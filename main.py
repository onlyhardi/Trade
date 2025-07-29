import logging
import os
import re
import time
import json
import base64
import base58
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from mnemonic import Mnemonic

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler
)

# Core Solana types, moved to solders for performance and consistency
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction, VersionedTransaction # <--- Make sure this is correct now!
from solders.system_program import transfer # For SystemProgram instructions like SOL transfer
from solders.hash import Hash # Often needed for recent_blockhash
from solders.message import Message, MessageV0 # For building transaction messages

# Solana.py RPC client and related types (these should largely remain in solana.*)
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts, TokenAccountOpts
from solana.rpc.api import RPCException # For handling RPC errors
from solana.rpc.commitment import Confirmed # For transaction confirmation
from solana.transaction import AccountMeta, TransactionInstruction # For general instructions if needed

# SPL Token related imports (these are usually fine in spl.token.*)
from spl.token.async_client import AsyncToken
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import (
    get_associated_token_address,
    revoke,
    RevokeParams,
    create_associated_token_account,
    get_mint
)

# If you are using jupiter-python-sdk, keep its import
from jupiter_python_sdk.jupiter import Jupiter

# --- Load environment variables ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

# --- Global state ---
_user_wallets = {} # {user_id: {'wallets': [Keypair, ...], 'current_index': int}}

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Conversation states ---
WALLET_IMPORT = 1
BUY_TOKEN_ADDRESS = 2
BUY_TOKEN_AMOUNT = 3
SELL_TOKEN_ADDRESS = 4
SELL_TOKEN_AMOUNT = 5
CREATE_TOKEN_NAME = 6
CREATE_TOKEN_SYMBOL = 7
CREATE_TOKEN_DECIMALS = 8
CREATE_TOKEN_SUPPLY = 9
TRANSFER_SOL_RECIPIENT = 10
TRANSFER_SOL_AMOUNT = 11
REVOKE_TOKEN_MINT = 12
REVOKE_TOKEN_OWNER = 13 # This state might be redundant if owner is always current wallet
SWAP_CONFIRMATION = 14
REVOKE_CONFIRMATION = 15

# --- Constants ---
SOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")
DEFAULT_SLIPPAGE_BPS = 50  # 0.5% slippage

# --- Rate Limiting ---
user_last_command_time = {}
RATE_LIMIT_SECONDS = 1.5

def is_rate_limited(user_id: int) -> bool:
    """Checks if a user is rate-limited."""
    current_time = time.time()
    if user_id in user_last_command_time:
        if (current_time - user_last_command_time[user_id]) < RATE_LIMIT_SECONDS:
            return True
    user_last_command_time[user_id] = current_time
    return False

# --- Caching for SOL price ---
SOL_PRICE_CACHE = {"price": None, "timestamp": None}
CACHE_DURATION_SECONDS = 45

async def get_cached_sol_price():
    """Fetches SOL price, using cache if available and fresh."""
    current_time = datetime.now()
    if SOL_PRICE_CACHE["price"] and SOL_PRICE_CACHE["timestamp"] and \
       (current_time - SOL_PRICE_CACHE["timestamp"]) < timedelta(seconds=CACHE_DURATION_SECONDS):
        logger.debug("Using cached SOL price.")
        return SOL_PRICE_CACHE["price"]
    else:
        logger.info("Fetching new SOL price.")
        price = await _get_sol_price()
        if price is not None:
            SOL_PRICE_CACHE["price"] = price
            SOL_PRICE_CACHE["timestamp"] = current_time
        return price

async def _get_sol_price() -> float | None:
    """Fetches the current SOL price in USD from CoinGecko API."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
        logger.info(f"Fetching SOL price from CoinGecko: {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "solana" in data and "usd" in data["solana"]:
            price = data["solana"]["usd"]
            logger.info(f"Successfully fetched SOL price: {price}")
            return float(price)
        else:
            logger.warning("SOL price not found in CoinGecko response.")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching SOL price: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching SOL price: {e}")
        return None

# --- Jupiter DEX Aggregator Functions ---
async def get_jupiter_client(payer_keypair: Keypair = None) -> Jupiter:
    """Initializes and returns a Jupiter client."""
    # The AsyncClient should ideally be managed as a singleton or within a context manager
    # for persistent connections, but for simple bot requests, creating it per call is okay.
    # For production, consider using a single client instance or a connection pool.
    async_client = AsyncClient(SOLANA_RPC_URL)
    # Jupiter client expects a Keypair for signing, even if just for quote
    # It will use the payer_keypair for swap.
    jupiter = Jupiter(async_client, payer_keypair)
    return jupiter

async def get_swap_quote(
    input_mint: Pubkey,
    output_mint: Pubkey,
    amount: int, # amount in lamports for SOL, or smallest unit for tokens
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
) -> dict | None:
    """Fetches a swap quote from Jupiter Aggregator."""
    jupiter_client = await get_jupiter_client() # No payer keypair needed for quote
    try:
        quote = await jupiter_client.quote(
            input_mint=str(input_mint),
            output_mint=str(output_mint),
            amount=amount,
            slippage_bps=slippage_bps
        )
        logger.info(f"Jupiter Quote received: {json.dumps(quote, indent=2)}")
        return quote
    except Exception as e:
        logger.error(f"Error fetching Jupiter quote for {input_mint} to {output_mint} amount {amount}: {e}", exc_info=True)
        return None
    finally:
        # It's good practice to close the client if it's created per request
        if jupiter_client._solana_client:
            await jupiter_client._solana_client.close()


async def execute_jupiter_swap(
    payer_keypair: Keypair,
    quote_response: dict # The full quote object received from get_swap_quote
) -> str | None:
    """Executes a Jupiter swap given a quote response."""
    jupiter_client = await get_jupiter_client(payer_keypair)
    try:
        # Get swap transaction from the quote
        swap_tx_serialized = await jupiter_client.swap(quote_response=quote_response)

        # Deserialize the transaction, sign and send
        # The swap_tx_serialized is a base64 string, representing a VersionedTransaction
        # Jupiter provides the transaction ready to be sent
        tx_bytes = base64.b64decode(swap_tx_serialized['swapTransaction'])
        transaction = VersionedTransaction.from_bytes(tx_bytes)

        # The transaction from Jupiter is usually partially signed or ready to be signed by the payer
        # Check if it needs to be signed by the payer. Jupiter's SDK should handle this.
        # If the transaction is already fully signed, this step is not strictly needed.
        # If it's a message, it needs to be wrapped in a transaction and signed.
        # The 'swap' method in jupiter-python-sdk returns the serialized transaction.
        # We need to ensure it's properly signed by the payer before sending.

        # The `send_raw_transaction` method expects a signed transaction.
        # If the `swap_tx_serialized` already includes the signature, this is fine.
        # If not, you'd need to manually sign:
        # transaction.sign([payer_keypair]) # This depends on the exact structure Jupiter returns.

        async with AsyncClient(SOLANA_RPC_URL) as client:
            opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
            result = await client.send_raw_transaction(tx_bytes, opts=opts) # Send the raw bytes
            txn_id = result.value
            logger.info(f"Swap transaction sent: {txn_id}")
            # Wait for confirmation
            await client.confirm_transaction(txn_id, Confirmed)
            logger.info(f"Swap transaction confirmed: {txn_id}")
            return str(txn_id)

    except RPCException as e:
        logger.error(f"Solana RPC error executing Jupiter swap: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error executing Jupiter swap: {e}", exc_info=True)
        return None
    finally:
        if jupiter_client._solana_client:
            await jupiter_client._solana_client.close()

async def get_token_decimals(mint_address: Pubkey) -> int | None:
    """Fetches the decimals for a token."""
    try:
        async with AsyncClient(SOLANA_RPC_URL) as client:
            mint_info = await get_mint(client, mint_address)
            return mint_info.decimals
    except RPCException as e:
        if "Account not found" in str(e):
            logger.warning(f"Token mint {mint_address} not found on chain: {e}")
            return None
        logger.error(f"RPC error fetching token decimals for {mint_address}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching token decimals for {mint_address}: {e}")
        return None

async def get_token_symbol_and_name(mint_address: Pubkey) -> tuple[str, str]:
    """Fetches token symbol and name using a common token list or metadata if available.
    For simplicity, we'll use a placeholder or basic naming if not found."""
    # In a real bot, you'd integrate with a token metadata service or a local database
    # For now, we'll just return derived names
    try:
        # Attempt to get metadata from Solana's metaplex if possible, or a token list API
        # This is a placeholder for actual metadata fetching
        return f"TKN{str(mint_address)[-4:]}", f"Token {str(mint_address)[:6]}..."
    except Exception:
        return f"TKN{str(mint_address)[-4:]}", f"Token {str(mint_address)[:6]}..."

# --- Wallet Manager Functions ---
def _import_wallet_from_phrase(phrase: str) -> Keypair | None:
    """Imports a Solana Keypair from a 12 or 24-word mnemonic phrase."""
    try:
        mnemonic = Mnemonic("english")
        if not mnemonic.check(phrase):
            logger.warning("Invalid mnemonic phrase provided.")
            return None
        seed = mnemonic.to_seed(phrase, passphrase="")
        return Keypair.from_seed(seed[:32])
    except Exception as e:
        logger.error(f"Error importing wallet from phrase: {e}")
        return None

def _import_wallet_from_privkey(privkey_base58: str) -> Keypair | None:
    """Imports a Solana Keypair from a base58 encoded private key."""
    try:
        secret = base58.b58decode(privkey_base58)
        if len(secret) == 64: # A 64-byte secret key is typically a 32-byte seed + 32-byte public key
            return Keypair.from_secret_key(secret)
        elif len(secret) == 32: # A 32-byte seed
            return Keypair.from_seed(secret)
        else:
            logger.warning(f"Invalid private key length: {len(secret)} bytes. Expected 32 or 64.")
            return None
    except ValueError:
        logger.warning("Invalid base58 private key format.")
        return None
    except Exception as e:
        logger.error(f"Error importing wallet from private key: {e}")
        return None

def _get_user_wallets(user_id: int) -> dict:
    """Retrieves the wallet data for a specific user."""
    if user_id not in _user_wallets:
        _user_wallets[user_id] = {'wallets': [], 'current_index': 0}
    return _user_wallets[user_id]

def _get_current_wallet(user_id: int) -> Keypair | None:
    """Gets the currently active wallet for a user."""
    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)
    if wallets and 0 <= current_index < len(wallets):
        return wallets[current_index]
    return None

def _add_wallet_to_user(user_id: int, keypair: Keypair):
    """Adds a new wallet to a user's list of wallets."""
    user_data = _get_user_wallets(user_id)
    if any(kp.pubkey() == keypair.pubkey() for kp in user_data['wallets']):
        logger.info(f"Wallet {keypair.pubkey()} already exists for user {user_id}.")
        return False
    user_data['wallets'].append(keypair)
    if len(user_data['wallets']) == 1: # If it's the first wallet, make it current
        user_data['current_index'] = 0
    logger.info(f"Wallet {keypair.pubkey()} added for user {user_id}.")
    return True

def _remove_wallet_from_user(user_id: int, index: int) -> bool:
    """Removes a wallet at a specific index for a user."""
    user_data = _get_user_wallets(user_id)
    if 0 <= index < len(user_data['wallets']):
        removed_wallet_pubkey = user_data['wallets'][index].pubkey()
        del user_data['wallets'][index]
        logger.info(f"Wallet {removed_wallet_pubkey} removed for user {user_id}.")
        if not user_data['wallets']:
            user_data['current_index'] = 0
        elif user_data['current_index'] >= len(user_data['wallets']):
            user_data['current_index'] = len(user_data['wallets']) - 1 # Adjust index if the current one was removed
        return True
    logger.warning(f"Attempted to remove non-existent wallet at index {index} for user {user_id}.")
    return False

def _set_current_wallet_index(user_id: int, index: int) -> bool:
    """Sets the currently active wallet by index for a user."""
    user_data = _get_user_wallets(user_id)
    if 0 <= index < len(user_data['wallets']):
        user_data['current_index'] = index
        logger.info(f"User {user_id} current wallet index set to {index}.")
        return True
    logger.warning(f"Attempted to set invalid current wallet index {index} for user {user_id}.")
    return False

# --- Solana Utils Functions ---
async def _get_sol_balance(rpc_url: str, pubkey_str: str) -> float:
    """Fetches the SOL balance for a given public key."""
    try:
        async with AsyncClient(rpc_url) as client:
            res = await client.get_balance(Pubkey.from_string(pubkey_str))
            lamports = res.value
            return lamports / 1e9
    except RPCException as e:
        logger.error(f"Solana RPC error fetching SOL balance for {pubkey_str}: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching SOL balance for {pubkey_str}: {e}")
        return 0.0

async def _request_devnet_airdrop(pubkey_str: str) -> str | None:
    """Requests a SOL airdrop on Devnet for a given public key."""
    devnet_rpc_url = "https://api.devnet.solana.com"
    try:
        async with AsyncClient(devnet_rpc_url) as client:
            res = await client.request_airdrop(Pubkey.from_string(pubkey_str), 1_000_000_000) # 1 SOL
            signature = res.value
            if signature:
                # Wait for confirmation
                try:
                    await client.confirm_transaction(signature, commitment='confirmed')
                    logger.info(f"Devnet airdrop confirmed: {signature}")
                    return signature
                except Exception as confirm_e:
                    logger.warning(f"Airdrop transaction {signature} not confirmed yet: {confirm_e}")
                    return signature # Still return signature, but warn about confirmation
            else:
                logger.warning(f"Airdrop request returned no signature: {res}")
                return None
    except RPCException as e:
        logger.error(f"Solana RPC error during devnet airdrop for {pubkey_str}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error requesting devnet airdrop for {pubkey_str}: {e}")
        return None

# --- Token Tools Functions ---
async def _create_spl_token_mint(
    rpc_url: str,
    payer: Keypair,
    decimals: int,
) -> tuple[Pubkey, str]: # Returns mint_pubkey, transaction_signature
    """Creates a new SPL token mint account."""
    async with AsyncClient(rpc_url) as client:
        # Generate a new Keypair for the mint itself
        mint_keypair = Keypair()

        # Create mint account
        create_mint_tx = await AsyncToken.create_mint(
            conn=client,
            payer=payer,
            mint_authority=payer.pubkey(), # Payer is mint authority initially
            freeze_authority=payer.pubkey(), # Payer is freeze authority initially
            decimals=decimals,
            program_id=TOKEN_PROGRAM_ID,
            skip_confirmation=False,
            opts=TxOpts(skip_preflight=False),
            mint_keypair=mint_keypair # Provide the mint keypair
        )
        tx_signature = create_mint_tx.value # The signature of the create mint transaction
        mint_pubkey = mint_keypair.pubkey()
        logger.info(f"Created SPL Token Mint: {mint_pubkey} with Tx: {tx_signature}")
        return mint_pubkey, tx_signature

async def _mint_spl_tokens(
    rpc_url: str,
    payer: Keypair,
    mint_pubkey: Pubkey,
    amount: int, # amount in base units (e.g., 1_000_000 for 1 token with 6 decimals)
) -> str: # Returns transaction_signature
    """Mints new SPL tokens to the payer's associated token account."""
    async with AsyncClient(rpc_url) as client:
        token_client = AsyncToken(client, mint_pubkey, TOKEN_PROGRAM_ID, payer)
        owner_pubkey = payer.pubkey()
        associated_token_address = get_associated_token_address(owner_pubkey, mint_pubkey)

        # Check if ATA exists, create if not
        try:
            await client.get_account_info(associated_token_address)
            logger.info(f"Associated Token Account {associated_token_address} already exists for {owner_pubkey}.")
        except RPCException as e:
            if "Account not found" in str(e): # Specific check for account not found
                logger.info(f"Creating Associated Token Account {associated_token_address} for {owner_pubkey}.")
                create_ata_ix = create_associated_token_account(
                    payer=owner_pubkey,
                    owner=owner_pubkey,
                    mint=mint_pubkey,
                    program_id=TOKEN_PROGRAM_ID
                )
                recent_blockhash = (await client.get_latest_blockhash()).value.blockhash
                transaction = Transaction(recent_blockhash=recent_blockhash)
                transaction.add(create_ata_ix)
                transaction.sign([payer])
                create_ata_sig = await client.send_transaction(transaction, opts=TxOpts(skip_preflight=False))
                await client.confirm_transaction(create_ata_sig.value, commitment='confirmed')
                logger.info(f"ATA created with signature: {create_ata_sig.value}")
            else:
                raise e # Re-raise if it's another RPC error

        tx_signature = await token_client.mint_to(
            dest=associated_token_address,
            owner=payer, # The mint authority, which is `payer` in this context
            amount=amount,
            multi_signers=[], # No multi-signers for this simple case
            opts=TxOpts(skip_preflight=False) # Ensure preflight for better error reporting
        )
        logger.info(f"Minted {amount} tokens to {associated_token_address} with Tx: {tx_signature}")
        return tx_signature

async def _get_token_accounts_and_balances(rpc_url: str, owner_pubkey: Pubkey) -> dict:
    """Fetches all SPL token accounts and their balances for a given owner."""
    token_balances = {}
    async with AsyncClient(rpc_url) as client:
        try:
            response = await client.get_token_accounts_by_owner(
                owner_pubkey,
                TokenAccountOpts(program_id=TOKEN_PROGRAM_ID),
                commitment='confirmed'
            )
            accounts = response.value

            for account in accounts:
                pubkey = account.pubkey
                # The data structure for parsed account info can vary slightly, adjust as needed
                account_info = account.account.data.parsed['info']
                mint_address = Pubkey.from_string(account_info['mint'])
                token_amount = account_info['tokenAmount']
                amount = float(token_amount['uiAmountString'])
                decimals = token_amount['decimals']

                symbol, name = await get_token_symbol_and_name(mint_address)

                token_balances[str(mint_address)] = { # Use string key for dictionary
                    'amount': amount,
                    'decimals': decimals,
                    'name': name,
                    'symbol': symbol,
                    'token_account': str(pubkey)
                }
        except RPCException as e:
            logger.error(f"Solana RPC error fetching token accounts for {owner_pubkey}: {e}")
        except Exception as e:
            logger.error(f"Error fetching token accounts for {owner_pubkey}: {e}")
    return token_balances

async def _revoke_token_delegate(
    rpc_url: str,
    owner_wallet: Keypair,
    token_account_pubkey: Pubkey # The specific token account to revoke delegate from
) -> str:
    """Revokes any delegate authority on a specific token account."""
    async with AsyncClient(rpc_url) as client:
        revoke_ix = revoke(RevokeParams(
            program_id=TOKEN_PROGRAM_ID,
            account=token_account_pubkey,
            owner=owner_wallet.pubkey(),
            signers=[] # No multi-signers for this simple case
        ))
        recent_blockhash = (await client.get_latest_blockhash()).value.blockhash
        transaction = Transaction(recent_blockhash=recent_blockhash)
        transaction.add(revoke_ix)
        try:
            signature_resp = await client.send_transaction(transaction, owner_wallet, opts=TxOpts(skip_preflight=False))
            signature = signature_resp.value # signature_resp.value contains the txid
            logger.info(f"Revoke delegate for {token_account_pubkey} with Tx: {signature}")
            await client.confirm_transaction(signature, commitment='confirmed')
            return signature
        except RPCException as e:
            logger.error(f"RPC error revoking delegate for {token_account_pubkey}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error revoking delegate for {token_account_pubkey}: {e}")
            raise e

# --- Helper functions for UI messages ---
async def get_start_message(user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    """Generates the /start message based on wallet connection status."""
    sol_price = await get_cached_sol_price()
    price_str = f"${sol_price:.2f}" if sol_price is not None else "unavailable"

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_wallet_index = user_data.get('current_index', 0)

    msg_parts = [
        "🚀 Official Pumpfun trading bot: Your Gateway to Solana DeFi 🔫\n",
        f"💰 SOL Price: {price_str}\n"
    ]

    if wallets and current_wallet_index < len(wallets):
        wallet = wallets[current_wallet_index]
        sol_balance = await _get_sol_balance(SOLANA_RPC_URL, str(wallet.pubkey()))
        usd_val = sol_balance * sol_price if sol_price is not None else 0

        msg_parts.append(
            "💳 Your Current Wallet:\n"
            f"↳ `{wallet.pubkey()}` 🅴\n"
            f"↳ SOL Balance: {sol_balance:.4f} SOL\n"
            f"↳ Total USD : ${usd_val:.2f}\n"
        )
    else:
        msg_parts.append(
            "💳 Connect your first wallet at /wallets\n"
        )
    msg_parts.append("Telegram | Twitter | Website")
    msg = "".join(msg_parts)

    keyboard = [
        [InlineKeyboardButton("🚀 Buy & Sell", callback_data="buy_sell"), InlineKeyboardButton("📍 Token Sniper", callback_data="sniper")],
        [InlineKeyboardButton("🎯 Sniper Pumpfun", callback_data="pumpfun"), InlineKeyboardButton("📍 Sniper Moonshot", callback_data="moonshot")],
        [InlineKeyboardButton("✒️ Limit Orders", callback_data="limit_orders")],
        [InlineKeyboardButton("🐒 Profile", callback_data="profile"), InlineKeyboardButton("💼 Wallets", callback_data="wallets_menu"), InlineKeyboardButton("📊 Trades", callback_data="trades")],
        [InlineKeyboardButton("🎮 Copy Trades", callback_data="copy"), InlineKeyboardButton("🎫 Referral System", callback_data="referral")],
        [InlineKeyboardButton("💸 Transfer SOL", callback_data="transfer_sol"), InlineKeyboardButton("🛠 Settings", callback_data="settings")],
        [InlineKeyboardButton("🔥 Our STBOT Tools", callback_data="tools"), InlineKeyboardButton("🚀 Market Maker Bot", callback_data="market_maker")],
        [InlineKeyboardButton("🧊 Backup Bots", callback_data="backup"), InlineKeyboardButton("🛡 Security", callback_data="security")],
        [InlineKeyboardButton("ℹ️ Help", callback_data="help"), InlineKeyboardButton("📄 Tutorials", callback_data="tutorials")],
        [InlineKeyboardButton("❌ Close", callback_data="close")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return msg, reply_markup

# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    msg, reply_markup = await get_start_message(user_id)
    await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")

async def wallets_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /wallets command."""
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if wallets and current_index < len(wallets):
        wallet = wallets[current_index]
        sol_balance = await _get_sol_balance(SOLANA_RPC_URL, str(wallet.pubkey()))
        sol_price = await get_cached_sol_price() or 0
        usd_val = sol_balance * sol_price

        msg = (
            "💳 Your Current Wallet:\n"
            f"Address: `{wallet.pubkey()}` 🅴\n"
            f"SOL Balance: {sol_balance:.4f} SOL\n"
            f"Estimated USD: ${usd_val:.2f}\n\n"
            "Use /wallets_menu to manage your wallets."
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    else:
        await update.message.reply_text(
            "You have no wallet connected.\n"
            "Type /importwallet to import one with your secret phrase or private key."
        )

async def wallets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays the wallet management menu."""
    user_id = update.callback_query.from_user.id if update.callback_query else update.effective_user.id
    if is_rate_limited(user_id):
        target_message = update.callback_query.message if update.callback_query else update.message
        await target_message.reply_text("Please wait a moment before sending another command.")
        return

    if update.callback_query:
        await update.callback_query.answer()

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    msg = "💼 *Wallet Management*\n\n"
    if not wallets:
        msg += "You have no wallets connected. Use /importwallet to add one.\n"
    else:
        msg += "Your connected wallets:\n"
        for i, wallet in enumerate(wallets):
            status = "✅ (Current)" if i == current_index else ""
            msg += f"  `{str(wallet.pubkey())[:6]}...{str(wallet.pubkey())[-4:]}` {status}\n"
        msg += "\nSelect a wallet to switch, or use the options below:\n"

    keyboard = []
    for i, wallet in enumerate(wallets):
        keyboard.append([InlineKeyboardButton(f"Switch to Wallet {i+1} ({str(wallet.pubkey())[:6]}...)", callback_data=f"switch_wallet_{i}")])

    keyboard.append([InlineKeyboardButton("➕ Add New Wallet", callback_data="add_wallet")])
    if wallets:
        keyboard.append([InlineKeyboardButton("➖ Remove Current Wallet", callback_data="remove_current_wallet")])
        keyboard.append([InlineKeyboardButton("🔑 Export Private Key (Current)", callback_data="export_privkey")])
        keyboard.append([InlineKeyboardButton("💰 Show All Token Balances", callback_data="show_token_balances")])
        keyboard.append([InlineKeyboardButton("🚫 Revoke Token Permissions", callback_data="revoke_permissions")])

    keyboard.append([InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")

async def switch_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switches the current active wallet for the user."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    try:
        index_to_switch = int(query.data.split('_')[-1])
        user_data = _get_user_wallets(user_id)
        wallets = user_data.get('wallets', [])

        if 0 <= index_to_switch < len(wallets):
            _set_current_wallet_index(user_id, index_to_switch)
            await query.edit_message_text(f"✅ Switched to wallet {index_to_switch + 1}.\n"
                                          f"Current wallet: `{wallets[index_to_switch].pubkey()}`", parse_mode="Markdown")
            msg, reply_markup = await get_start_message(user_id)
            # Send new message with updated main menu
            await query.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await query.edit_message_text("❌ Invalid wallet index.")
    except Exception as e:
        logger.error(f"Error switching wallet: {e}")
        await query.edit_message_text("❌ An error occurred while switching wallets.")

async def add_wallet_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Prompts the user to add a new wallet."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    await query.edit_message_text(
        "⚠️ *WARNING*: \n"
        "Do not share to public group.\n\n"
        "Send your 12/24-word secret phrase OR your base58 private key string now.\n"
        "Or /cancel to abort."
    , parse_mode="Markdown")
    return WALLET_IMPORT

async def remove_current_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Removes the currently active wallet."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets:
        await query.edit_message_text("❌ No wallets to remove.")
        return

    if _remove_wallet_from_user(user_id, current_index):
        await query.edit_message_text("✅ Current wallet removed successfully.")
        await wallets_menu(update, context) # Go back to wallet menu
    else:
        await query.edit_message_text("❌ Failed to remove wallet.")

async def export_private_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Exports the private key of the currently active wallet."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    wallet = _get_current_wallet(user_id)

    if not wallet:
        await query.edit_message_text("❌ No wallet connected to export.")
        return

    private_key_base58 = base58.b58encode(wallet.secret_key).decode('utf-8')

    await query.edit_message_text(
        "⚠️ *DANGER: Your Private Key!* ⚠️\n\n"
        "This key grants full access to your funds.*\n"
        "Save it securely and do not share to public groups.\n\n"
        f"`{private_key_base58}`\n\n"
        "Type /wallets_menu to go back."
    , parse_mode="Markdown")

async def show_token_balances(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays all SPL token balances for the current wallet."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    wallet = _get_current_wallet(user_id)

    if not wallet:
        await query.edit_message_text("❌ No wallet connected to view token balances.")
        return

    await query.edit_message_text(f"Fetching token balances for `{wallet.pubkey()}`...", parse_mode="Markdown")

    try:
        token_balances = await _get_token_accounts_and_balances(SOLANA_RPC_URL, wallet.pubkey())
        if not token_balances:
            msg = "No SPL tokens found in this wallet."
        else:
            msg = "📊 *Your Token Balances:*\n\n"
            for mint_address, details in token_balances.items():
                formatted_amount = f"{details['amount']:.{details['decimals']}f}".rstrip('0').rstrip('.')
                msg += (
                    f"*{details['symbol']}* ({details['name']})\n"
                    f"  Balance: {formatted_amount}\n"
                    f"  Mint: `{mint_address}`\n"
                    f"  Token Account: `{details['token_account']}`\n\n"
                )
        keyboard = [[InlineKeyboardButton("⬅️ Back to Wallet Menu", callback_data="wallets_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error fetching token balances: {e}")
        await query.edit_message_text("❌ Failed to fetch token balances. Please try again later.")

async def import_wallet_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the conversation for importing a wallet."""
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    await update.message.reply_text(
        "⚠️ *WARNING*: Do not share secret phrase or private key to public chats.\n"
        "These grant full access to your funds. Only use test wallets.\n\n"
        "Send your 12/24-word secret phrase OR your base58 private key string now.\n"
        "Or /cancel to abort."
    , parse_mode="Markdown")
    return WALLET_IMPORT

async def wallet_import_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the secret phrase or private key and attempts to import the wallet."""
    user_id = update.message.from_user.id
    text = update.message.text.strip()

    wallet = _import_wallet_from_phrase(text)
    if wallet is None:
        wallet = _import_wallet_from_privkey(text)

    if wallet is not None:
        if _add_wallet_to_user(user_id, wallet):
            await update.message.reply_text(
                f"✅ Wallet imported successfully.\nAddress: `{wallet.pubkey()}`\n⚠️ Only use test wallets.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                f"ℹ️ Wallet `{wallet.pubkey()}` was already connected. No new wallet added.",
                parse_mode="Markdown"
            )
        msg, reply_markup = await get_start_message(user_id)
        await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
        return ConversationHandler.END
    else:
        await update.message.reply_text(
            "❌ Failed to import wallet. Please make sure you sent a valid 12/24-word secret phrase or base58 private key."
        )
        return WALLET_IMPORT

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancels the current conversation."""
    if update.message:
        await update.message.reply_text("Operation canceled.")
    elif update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.edit_text("Operation canceled.")
    context.user_data.clear() # Clear user-specific conversation data
    return ConversationHandler.END

async def token_info_detector(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Detects if a message contains a potential Solana token address (Pubkey)
    and offers options to buy/sell that token.
    """
    message_text = update.message.text
    user_id = update.effective_user.id

    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    # Regex to find a potential base58 encoded Pubkey (32-44 characters long)
    # Solana Pubkeys are typically 32-44 base58 characters.
    pubkey_pattern = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
    found_pubkeys = re.findall(pubkey_pattern, message_text)

    for pubkey_str in found_pubkeys:
        try:
            potential_mint = Pubkey.from_string(pubkey_str)
            # Basic check if it's a valid Pubkey format
            # Further validation (e.g., checking if it's an actual mint) will be done when fetching decimals

            context.user_data['detected_token_mint'] = str(potential_mint) # Store as string
            symbol, name = await get_token_symbol_and_name(potential_mint) # Fetch metadata if possible

            keyboard = [
                [InlineKeyboardButton(f"Buy {symbol}", callback_data=f"buy_detected_token_{potential_mint}"),
                 InlineKeyboardButton(f"Sell {symbol}", callback_data=f"sell_detected_token_{potential_mint}")],
                [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                f"Looks like you mentioned a token address: `{pubkey_str}`.\n"
                f"Would you like to trade *{name}* ({symbol})?",
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
            return
        except Exception as e:
            logger.debug(f"Not a valid Solana Pubkey or other error: {pubkey_str}, error: {e}")
            # Continue to check other potential pubkeys or ignore if not a valid one
            continue
    # If no valid token address is detected, let other handlers process the message
    return

# --- Buy/Sell Token Handlers ---
async def buy_sell_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays the main buy/sell menu."""
    user_id = update.callback_query.from_user.id if update.callback_query else update.effective_user.id
    if is_rate_limited(user_id):
        await (update.callback_query.message if update.callback_query else update.message).reply_text("Please wait a moment before sending another command.")
        return
    if update.callback_query:
        await update.callback_query.answer()

    keyboard = [
        [InlineKeyboardButton("📈 Buy Token", callback_data="start_buy_token")],
        [InlineKeyboardButton("📉 Sell Token", callback_data="start_sell_token")],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    msg = "💰 *Buy & Sell Tokens*\n\nChoose an option below to start trading."
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
    else:
        await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")

async def start_buy_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the process for buying a token."""
    user_id = update.callback_query.from_user.id
    if is_rate_limited(user_id):
        await update.callback_query.message.reply_text("Please wait a moment before sending another command.")
        return
    await update.callback_query.answer()

    wallet = _get_current_wallet(user_id)
    if not wallet:
        await update.callback_query.edit_message_text("❌ No wallet connected. Please connect a wallet first using /importwallet or from the Wallets menu.")
        return ConversationHandler.END

    await update.callback_query.edit_message_text(
        "📝 Please send the *token address (mint address)* of the token you wish to buy.\n"
        "Or /cancel to abort."
    , parse_mode="Markdown")
    context.user_data['transaction_type'] = 'buy'
    return BUY_TOKEN_ADDRESS

async def start_sell_token(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the process for selling a token."""
    user_id = update.callback_query.from_user.id
    if is_rate_limited(user_id):
        await update.callback_query.message.reply_text("Please wait a moment before sending another command.")
        return
    await update.callback_query.answer()

    wallet = _get_current_wallet(user_id)
    if not wallet:
        await update.callback_query.edit_message_text("❌ No wallet connected. Please connect a wallet first using /importwallet or from the Wallets menu.")
        return ConversationHandler.END

    await update.callback_query.edit_message_text(
        "📝 Please send the *token address (mint address)* of the token you wish to sell.\n"
        "Or /cancel to abort."
    , parse_mode="Markdown")
    context.user_data['transaction_type'] = 'sell'
    return SELL_TOKEN_ADDRESS

async def handle_buy_sell_token_address(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the token address input for buy/sell operations."""
    user_id = update.message.from_user.id
    token_address_str = update.message.text.strip()

    try:
        token_mint_pubkey = Pubkey.from_string(token_address_str)
        decimals = await get_token_decimals(token_mint_pubkey)

        if decimals is None:
            await update.message.reply_text(
                "❌ Invalid token address or could not retrieve token information. Please check the address and try again.\n"
                "Send the *token address* or /cancel to abort."
            )
            return (BUY_TOKEN_ADDRESS if context.user_data['transaction_type'] == 'buy' else SELL_TOKEN_ADDRESS)

        context.user_data['target_token_mint'] = str(token_mint_pubkey)
        context.user_data['target_token_decimals'] = decimals

        symbol, name = await get_token_symbol_and_name(token_mint_pubkey)
        context.user_data['target_token_symbol'] = symbol
        context.user_data['target_token_name'] = name

        if context.user_data['transaction_type'] == 'buy':
            await update.message.reply_text(
                f"Buying *{name}* ({symbol}).\n"
                f"How much *SOL* do you want to spend? (e.g., `0.1`, `1.5`)\n"
                "Or /cancel to abort."
            , parse_mode="Markdown")
            return BUY_TOKEN_AMOUNT
        else: # Sell
            # For selling, we first need to check the user's balance of that token
            wallet = _get_current_wallet(user_id)
            if not wallet:
                await update.message.reply_text("❌ No wallet connected. Cannot check token balance.")
                return ConversationHandler.END

            token_balances = await _get_token_accounts_and_balances(SOLANA_RPC_URL, wallet.pubkey())
            target_token_balance_info = token_balances.get(str(token_mint_pubkey))

            if not target_token_balance_info or target_token_balance_info['amount'] == 0:
                await update.message.reply_text(
                    f"❌ You do not hold any *{name}* ({symbol}) in your current wallet (`{wallet.pubkey()}`).\n"
                    "Please enter a different token address or /cancel."
                , parse_mode="Markdown")
                return SELL_TOKEN_ADDRESS # Stay in SELL_TOKEN_ADDRESS state
            else:
                context.user_data['user_token_balance'] = target_token_balance_info['amount']
                await update.message.reply_text(
                    f"Selling *{name}* ({symbol}).\n"
                    f"Your current balance: `{target_token_balance_info['amount']:.{decimals}f} {symbol}`\n"
                    f"How much *{symbol}* do you want to sell? (e.g., `100`, `0.5`)\n"
                    "Or /cancel to abort."
                , parse_mode="Markdown")
                return SELL_TOKEN_AMOUNT

    except ValueError:
        await update.message.reply_text(
            "❌ Invalid Solana address format. Please provide a valid token address (Pubkey).\n"
            "Send the *token address* or /cancel to abort."
        )
        return (BUY_TOKEN_ADDRESS if context.user_data['transaction_type'] == 'buy' else SELL_TOKEN_ADDRESS)
    except Exception as e:
        logger.error(f"Error handling token address: {e}", exc_info=True)
        await update.message.reply_text("❌ An unexpected error occurred. Please try again or /cancel.")
        return ConversationHandler.END


async def handle_buy_token_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the SOL amount input for buying a token."""
    user_id = update.message.from_user.id
    sol_amount_str = update.message.text.strip()

    try:
        sol_amount_float = float(sol_amount_str)
        if sol_amount_float <= 0:
            raise ValueError("Amount must be positive.")

        # Convert SOL to lamports
        sol_amount_lamports = int(sol_amount_float * 1e9)
        context.user_data['input_amount_lamports'] = sol_amount_lamports
        context.user_data['input_amount_ui'] = sol_amount_float # Store UI amount for display

        wallet = _get_current_wallet(user_id)
        if not wallet:
            await update.message.reply_text("❌ No wallet connected. Cannot proceed with transaction.")
            return ConversationHandler.END

        # Get SOL balance to check if sufficient
        wallet_sol_balance = await _get_sol_balance(SOLANA_RPC_URL, str(wallet.pubkey()))
        # Add a small buffer for transaction fees
        BUFFER_SOL = 0.0001 # 0.0001 SOL for fees
        if sol_amount_float + BUFFER_SOL > wallet_sol_balance:
            await update.message.reply_text(
                f"❌ Insufficient SOL balance. You have {wallet_sol_balance:.4f} SOL. "
                f"You need at least {sol_amount_float + BUFFER_SOL:.4f} SOL (including fees).\n"
                "Please enter a smaller amount or /cancel to abort."
            )
            return BUY_TOKEN_AMOUNT

        input_mint = SOL_MINT
        output_mint = Pubkey.from_string(context.user_data['target_token_mint'])

        # Get Jupiter quote
        await update.message.reply_text(f"Fetching best swap quote for {sol_amount_float} SOL...")
        quote = await get_swap_quote(input_mint, output_mint, sol_amount_lamports)

        if quote:
            context.user_data['jupiter_quote'] = quote
            out_amount_lamports = quote['outAmount']
            output_token_decimals = context.user_data['target_token_decimals']
            out_amount_ui = out_amount_lamports / (10**output_token_decimals)
            estimated_price_per_token = sol_amount_float / out_amount_ui if out_amount_ui > 0 else float('inf')


            msg = (
                f"🔥 *Swap Confirmation (Buy)* 🔥\n\n"
                f"You are buying *{context.user_data['target_token_name']}* ({context.user_data['target_token_symbol']}).\n"
                f"Spending: `{sol_amount_float:.4f} SOL`\n"
                f"Receiving (est.): `{out_amount_ui:.{output_token_decimals}f} {context.user_data['target_token_symbol']}`\n"
                f"Price per {context.user_data['target_token_symbol']} (est.): `{estimated_price_per_token:.8f} SOL`\n"
                f"Slippage Tolerance: `{DEFAULT_SLIPPAGE_BPS / 100}%`\n\n"
                f"Confirm to proceed with the swap."
            )
            keyboard = [
                [InlineKeyboardButton("✅ Confirm Buy", callback_data="confirm_buy_swap")],
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
            return SWAP_CONFIRMATION
        else:
            await update.message.reply_text("❌ Could not get a swap quote. The token might not be tradable or there's insufficient liquidity. Please try again or /cancel.")
            return BUY_TOKEN_AMOUNT # Stay in the amount state
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid amount. Please enter a valid number for SOL (e.g., `0.5`, `10`).\n"
            "Or /cancel to abort."
        )
        return BUY_TOKEN_AMOUNT
    except Exception as e:
        logger.error(f"Error handling buy token amount: {e}", exc_info=True)
        await update.message.reply_text("❌ An unexpected error occurred. Please try again or /cancel.")
        return ConversationHandler.END

async def handle_sell_token_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the token amount input for selling a token."""
    user_id = update.message.from_user.id
    token_amount_str = update.message.text.strip()

    try:
        token_amount_float = float(token_amount_str)
        if token_amount_float <= 0:
            raise ValueError("Amount must be positive.")

        token_decimals = context.user_data['target_token_decimals']
        token_amount_smallest_unit = int(token_amount_float * (10**token_decimals))
        context.user_data['input_amount_smallest_unit'] = token_amount_smallest_unit
        context.user_data['input_amount_ui'] = token_amount_float # Store UI amount for display

        wallet = _get_current_wallet(user_id)
        if not wallet:
            await update.message.reply_text("❌ No wallet connected. Cannot proceed with transaction.")
            return ConversationHandler.END

        # Check if user has enough tokens to sell
        user_token_balance = context.user_data.get('user_token_balance', 0)
        if token_amount_float > user_token_balance:
            await update.message.reply_text(
                f"❌ Insufficient token balance. You have `{user_token_balance:.{token_decimals}f} {context.user_data['target_token_symbol']}`. "
                f"Please enter a smaller amount or /cancel to abort."
            , parse_mode="Markdown")
            return SELL_TOKEN_AMOUNT

        input_mint = Pubkey.from_string(context.user_data['target_token_mint'])
        output_mint = SOL_MINT

        # Get Jupiter quote
        await update.message.reply_text(f"Fetching best swap quote for {token_amount_float} {context.user_data['target_token_symbol']}...")
        quote = await get_swap_quote(input_mint, output_mint, token_amount_smallest_unit)

        if quote:
            context.user_data['jupiter_quote'] = quote
            out_amount_lamports = quote['outAmount']
            out_amount_ui = out_amount_lamports / 1e9 # SOL has 9 decimals
            estimated_price_per_token = out_amount_ui / token_amount_float if token_amount_float > 0 else 0

            msg = (
                f"🔥 *Swap Confirmation (Sell)* 🔥\n\n"
                f"You are selling *{context.user_data['target_token_name']}* ({context.user_data['target_token_symbol']}).\n"
                f"Selling: `{token_amount_float:.{token_decimals}f} {context.user_data['target_token_symbol']}`\n"
                f"Receiving (est.): `{out_amount_ui:.4f} SOL`\n"
                f"Price per {context.user_data['target_token_symbol']} (est.): `{estimated_price_per_token:.8f} SOL`\n"
                f"Slippage Tolerance: `{DEFAULT_SLIPPAGE_BPS / 100}%`\n\n"
                f"Confirm to proceed with the swap."
            )
            keyboard = [
                [InlineKeyboardButton("✅ Confirm Sell", callback_data="confirm_sell_swap")],
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
            return SWAP_CONFIRMATION
        else:
            await update.message.reply_text("❌ Could not get a swap quote. The token might not be tradable or there's insufficient liquidity. Please try again or /cancel.")
            return SELL_TOKEN_AMOUNT # Stay in the amount state
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid amount. Please enter a valid number for the token amount (e.g., `100`, `0.5`).\n"
            "Or /cancel to abort."
        )
        return SELL_TOKEN_AMOUNT
    except Exception as e:
        logger.error(f"Error handling sell token amount: {e}", exc_info=True)
        await update.message.reply_text("❌ An unexpected error occurred. Please try again or /cancel.")
        return ConversationHandler.END


async def confirm_swap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Executes the Jupiter swap after user confirmation."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    wallet = _get_current_wallet(user_id)
    if not wallet:
        await query.edit_message_text("❌ No wallet connected. Cannot execute swap.")
        context.user_data.clear()
        return ConversationHandler.END

    quote_response = context.user_data.get('jupiter_quote')
    transaction_type = context.user_data.get('transaction_type')
    input_amount_ui = context.user_data.get('input_amount_ui')
    target_token_symbol = context.user_data.get('target_token_symbol')

    if not quote_response:
        await query.edit_message_text("❌ Swap quote not found. Please restart the buy/sell process.")
        context.user_data.clear()
        return ConversationHandler.END

    await query.edit_message_text(f"🚀 Executing {transaction_type} of {input_amount_ui} {target_token_symbol or 'SOL'}... This may take a moment.")
    try:
        txn_id = await execute_jupiter_swap(wallet, quote_response)
        if txn_id:
            await query.edit_message_text(
                f"✅ Swap successful! 🎉\n"
                f"Transaction ID: [`{txn_id}`](https://solscan.io/tx/{txn_id})\n\n"
                "You can check your updated balances via the Wallet menu."
            , parse_mode="Markdown")
        else:
            await query.edit_message_text("❌ Swap failed. Please try again later or contact support.")
    except Exception as e:
        logger.error(f"Error during swap execution: {e}", exc_info=True)
        await query.edit_message_text(f"❌ An error occurred during the swap: {e}.\nPlease try again or contact support.")
    finally:
        context.user_data.clear() # Clear conversation data
        return ConversationHandler.END

# --- Create Token Handlers ---
async def create_token_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the create token conversation."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    wallet = _get_current_wallet(user_id)
    if not wallet:
        await query.edit_message_text("❌ No wallet connected. Please connect a wallet first to create a token.")
        return ConversationHandler.END

    await query.edit_message_text(
        "📝 Let's create your new SPL Token!\n"
        "First, please provide the *name* for your token (e.g., `My Awesome Token`).\n"
        "Or /cancel to abort."
    , parse_mode="Markdown")
    return CREATE_TOKEN_NAME

async def handle_create_token_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the token name input."""
    user_id = update.message.from_user.id
    token_name = update.message.text.strip()
    if not token_name:
        await update.message.reply_text("Token name cannot be empty. Please provide a name or /cancel.")
        return CREATE_TOKEN_NAME

    context.user_data['new_token_name'] = token_name
    await update.message.reply_text(
        f"Great! The token name will be: *{token_name}*.\n"
        "Now, please provide the *symbol* for your token (e.g., `MAT`).\n"
        "Or /cancel to abort."
    , parse_mode="Markdown")
    return CREATE_TOKEN_SYMBOL

async def handle_create_token_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the token symbol input."""
    user_id = update.message.from_user.id
    token_symbol = update.message.text.strip()
    if not token_symbol or len(token_symbol) > 10: # Common symbol length limit
        await update.message.reply_text("Token symbol cannot be empty and should be short (e.g., max 10 chars). Please provide a valid symbol or /cancel.")
        return CREATE_TOKEN_SYMBOL

    context.user_data['new_token_symbol'] = token_symbol
    await update.message.reply_text(
        f"Symbol: *{token_symbol}*.\n"
        "Next, what *decimals* should your token have? (e.g., `6` for standard, `9` for like SOL).\n"
        "Enter a number between 0 and 9. Or /cancel to abort."
    , parse_mode="Markdown")
    return CREATE_TOKEN_DECIMALS

async def handle_create_token_decimals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the token decimals input."""
    user_id = update.message.from_user.id
    decimals_str = update.message.text.strip()

    try:
        decimals = int(decimals_str)
        if not (0 <= decimals <= 9): # Standard range for decimals
            raise ValueError("Decimals must be between 0 and 9.")
        context.user_data['new_token_decimals'] = decimals
        await update.message.reply_text(
            f"Decimals set to: *{decimals}*.\n"
            "Finally, what is the *total supply* for your token? (e.g., `1000000` for 1 million tokens).\n"
            "This is the initial amount of tokens that will be minted to your wallet.\n"
            "Or /cancel to abort."
        , parse_mode="Markdown")
        return CREATE_TOKEN_SUPPLY
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid number for decimals. Please enter an integer between 0 and 9.\n"
            "Or /cancel to abort."
        )
        return CREATE_TOKEN_DECIMALS

async def handle_create_token_supply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the token total supply input and initiates token creation."""
    user_id = update.message.from_user.id
    total_supply_str = update.message.text.strip()

    try:
        total_supply = float(total_supply_str) # Allow float for user input, convert to int for actual supply
        if total_supply <= 0:
            raise ValueError("Total supply must be a positive number.")

        decimals = context.user_data['new_token_decimals']
        # Convert UI amount to smallest unit for minting
        total_supply_smallest_unit = int(total_supply * (10**decimals))
        if total_supply_smallest_unit == 0 and total_supply > 0: # Handle cases like 0.0000001 with high decimals becoming 0
             raise ValueError("The entered supply is too small given the decimals and would result in zero tokens.")

        wallet = _get_current_wallet(user_id)
        if not wallet:
            await update.message.reply_text("❌ No wallet connected. Cannot create token.")
            context.user_data.clear()
            return ConversationHandler.END

        token_name = context.user_data['new_token_name']
        token_symbol = context.user_data['new_token_symbol']

        await update.message.reply_text(f"Creating token *{token_name}* ({token_symbol}) with {decimals} decimals and a total supply of {total_supply:.{decimals}f}...")

        # Step 1: Create the mint account
        mint_pubkey, create_mint_tx_sig = await _create_spl_token_mint(
            SOLANA_RPC_URL,
            wallet,
            decimals,
        )
        await update.message.reply_text(
            f"✅ Token Mint Account created!\n"
            f"Mint Address: `{mint_pubkey}`\n"
            f"Creation Tx: [`{create_mint_tx_sig}`](https://solscan.io/tx/{create_mint_tx_sig})\n\n"
            "Now minting initial supply to your wallet..."
        , parse_mode="Markdown")

        # Step 2: Mint initial supply to the creator's associated token account
        mint_to_tx_sig = await _mint_spl_tokens(
            SOLANA_RPC_URL,
            wallet,
            mint_pubkey,
            total_supply_smallest_unit,
        )
        await update.message.reply_text(
            f"✅ Initial supply of {total_supply:.{decimals}f} *{token_symbol}* minted to your wallet!\n"
            f"Minting Tx: [`{mint_to_tx_sig}`](https://solscan.io/tx/{mint_to_tx_sig})\n\n"
            "Congratulations! Your token is live on Solana."
        , parse_mode="Markdown")

    except ValueError as ve:
        await update.message.reply_text(f"❌ Invalid amount for total supply: {ve}\nPlease enter a positive number or /cancel.")
        return CREATE_TOKEN_SUPPLY
    except RPCException as rpc_e:
        logger.error(f"RPC error creating/minting token: {rpc_e}", exc_info=True)
        await update.message.reply_text(f"❌ Solana network error during token creation: {rpc_e}.\nPlease try again later or check network status.")
    except Exception as e:
        logger.error(f"Error creating/minting token: {e}", exc_info=True)
        await update.message.reply_text("❌ An unexpected error occurred during token creation. Please try again or /cancel.")
    finally:
        context.user_data.clear()
        return ConversationHandler.END


# --- Transfer SOL Handlers ---
async def transfer_sol_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the SOL transfer process."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    wallet = _get_current_wallet(user_id)
    if not wallet:
        await query.edit_message_text("❌ No wallet connected. Please connect a wallet first to transfer SOL.")
        return ConversationHandler.END

    current_balance = await _get_sol_balance(SOLANA_RPC_URL, str(wallet.pubkey()))
    await query.edit_message_text(
        f"💸 *Transfer SOL*\n\n"
        f"Your current SOL balance: `{current_balance:.4f} SOL`\n"
        "Please send the *recipient's Solana address*.\n"
        "Or /cancel to abort."
    , parse_mode="Markdown")
    return TRANSFER_SOL_RECIPIENT

async def handle_transfer_sol_recipient(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the recipient address for SOL transfer."""
    user_id = update.message.from_user.id
    recipient_address_str = update.message.text.strip()

    try:
        recipient_pubkey = Pubkey.from_string(recipient_address_str)
        context.user_data['sol_recipient_pubkey'] = str(recipient_pubkey)

        await update.message.reply_text(
            f"Recipient set to: `{recipient_pubkey}`.\n"
            f"Now, how much *SOL* do you want to send? (e.g., `0.1`, `1.5`)\n"
            "Or /cancel to abort."
        , parse_mode="Markdown")
        return TRANSFER_SOL_AMOUNT
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid Solana address format for the recipient. Please provide a valid address.\n"
            "Send the *recipient's address* or /cancel to abort."
        )
        return TRANSFER_SOL_RECIPIENT
    except Exception as e:
        logger.error(f"Error handling SOL recipient: {e}", exc_info=True)
        await update.message.reply_text("❌ An unexpected error occurred. Please try again or /cancel.")
        return ConversationHandler.END

async def handle_transfer_sol_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the SOL amount and initiates the transfer."""
    user_id = update.message.from_user.id
    sol_amount_str = update.message.text.strip()

    try:
        sol_amount_float = float(sol_amount_str)
        if sol_amount_float <= 0:
            raise ValueError("Amount must be positive.")

        wallet = _get_current_wallet(user_id)
        if not wallet:
            await update.message.reply_text("❌ No wallet connected. Cannot complete transfer.")
            context.user_data.clear()
            return ConversationHandler.END

        # Check balance
        current_balance = await _get_sol_balance(SOLANA_RPC_URL, str(wallet.pubkey()))
        # Account for transaction fees (typically around 0.000005 SOL per simple transaction)
        # Add a small buffer for network fees
        transaction_fee_buffer = 0.00001 # 10_000 lamports for transfer
        if sol_amount_float + transaction_fee_buffer > current_balance:
            await update.message.reply_text(
                f"❌ Insufficient SOL balance. You have {current_balance:.4f} SOL. "
                f"You need at least {sol_amount_float + transaction_fee_buffer:.4f} SOL (including fees).\n"
                "Please enter a smaller amount or /cancel to abort."
            )
            return TRANSFER_SOL_AMOUNT

        recipient_pubkey = Pubkey.from_string(context.user_data['sol_recipient_pubkey'])
        amount_lamports = int(sol_amount_float * 1e9)

        await update.message.reply_text(f"Sending {sol_amount_float:.4f} SOL to `{recipient_pubkey}`...")

        async with AsyncClient(SOLANA_RPC_URL) as client:
            # Create a simple transfer instruction
            transfer_ix = transfer(
                TransferParams(
                    from_pubkey=wallet.pubkey(),
                    to_pubkey=recipient_pubkey,
                    lamports=amount_lamports
                )
            )

            recent_blockhash = (await client.get_latest_blockhash()).value.blockhash
            transaction = Transaction(recent_blockhash=recent_blockhash)
            transaction.add(transfer_ix)

            # Sign and send the transaction
            try:
                # The send_transaction method usually signs it internally with the provided keypair
                # if it's the fee payer or an instruction signer.
                # For a simple transfer, the payer is also the 'from_pubkey'.
                signature_resp = await client.send_transaction(transaction, wallet, opts=TxOpts(skip_preflight=False))
                tx_signature = signature_resp.value
                logger.info(f"SOL transfer transaction sent: {tx_signature}")

                await client.confirm_transaction(tx_signature, commitment='confirmed')
                await update.message.reply_text(
                    f"✅ SOL transfer successful! 🎉\n"
                    f"Sent `{sol_amount_float:.4f} SOL` to `{recipient_pubkey}`.\n"
                    f"Transaction ID: [`{tx_signature}`](https://solscan.io/tx/{tx_signature})"
                , parse_mode="Markdown")
            except RPCException as rpc_e:
                logger.error(f"RPC error during SOL transfer: {rpc_e}", exc_info=True)
                await update.message.reply_text(f"❌ Solana network error during transfer: {rpc_e}.\nPlease try again.")
            except Exception as e:
                logger.error(f"Error during SOL transfer: {e}", exc_info=True)
                await update.message.reply_text(f"❌ An unexpected error occurred during transfer: {e}.\nPlease try again or /cancel.")
    except ValueError as ve:
        await update.message.reply_text(f"❌ Invalid amount: {ve}\nPlease enter a valid number for SOL (e.g., `0.5`, `10`).\nOr /cancel to abort.")
        return TRANSFER_SOL_AMOUNT
    finally:
        context.user_data.clear()
        return ConversationHandler.END


# --- Revoke Token Permissions Handlers ---
async def revoke_permissions_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the revoke token permissions process."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    wallet = _get_current_wallet(user_id)
    if not wallet:
        await query.edit_message_text("❌ No wallet connected. Please connect a wallet first to revoke permissions.")
        return ConversationHandler.END

    # Fetch token accounts to list them for the user
    await query.edit_message_text(f"Fetching your token accounts to find those with delegates for `{wallet.pubkey()}`...")
    token_balances = await _get_token_accounts_and_balances(SOLANA_RPC_URL, wallet.pubkey())

    if not token_balances:
        await query.edit_message_text("ℹ️ No SPL token accounts found in your wallet. Nothing to revoke.")
        return ConversationHandler.END

    # Filter for accounts that might have a delegate (though `get_token_accounts_by_owner` doesn't directly show delegates)
    # The actual check for a delegate will happen during the revoke instruction simulation or execution.
    # For now, we'll list all token accounts and let the user pick one.
    msg = "🚫 *Revoke Token Permissions*\n\n" \
          "Select a token account from which you want to revoke delegate authority.\n\n" \
          "Note: Only token accounts that actually have a delegate set will have an effect.\n\n" \
          "*Your Token Accounts:*\n"
    keyboard = []
    has_tokens_to_list = False
    for mint_address, details in token_balances.items():
        if details['amount'] > 0: # Only list accounts with balance
            has_tokens_to_list = True
            token_account_pubkey = details['token_account']
            msg += (
                f"  *{details['symbol']}* ({details['name']})\n"
                f"  Balance: {details['amount']:.{details['decimals']}f}\n"
                f"  Account: `{token_account_pubkey}`\n\n"
            )
            keyboard.append([InlineKeyboardButton(f"Revoke {details['symbol']} ({token_account_pubkey[:6]}...)", callback_data=f"select_revoke_token_account_{token_account_pubkey}")])

    if not has_tokens_to_list:
        msg = "ℹ️ No SPL token accounts with a balance found in your wallet. Nothing to revoke."
        keyboard = [[InlineKeyboardButton("⬅️ Back to Wallet Menu", callback_data="wallets_menu")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
        return ConversationHandler.END

    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="cancel")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
    return REVOKE_TOKEN_MINT # Using this state to capture the selected token account pubkey

async def handle_revoke_token_account_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the selection of a token account to revoke permissions from."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    token_account_str = query.data.replace("select_revoke_token_account_", "")
    try:
        token_account_pubkey = Pubkey.from_string(token_account_str)
        context.user_data['revoke_target_token_account'] = str(token_account_pubkey)

        wallet = _get_current_wallet(user_id)
        if not wallet:
            await query.edit_message_text("❌ No wallet connected. Cannot proceed with revoke.")
            context.user_data.clear()
            return ConversationHandler.END

        # Get token details for confirmation message
        token_balances = await _get_token_accounts_and_balances(SOLANA_RPC_URL, wallet.pubkey())
        selected_token_info = None
        for mint, info in token_balances.items():
            if info['token_account'] == str(token_account_pubkey):
                selected_token_info = info
                break

        if not selected_token_info:
            await query.edit_message_text("❌ Could not find details for the selected token account. Please try again.")
            context.user_data.clear()
            return ConversationHandler.END

        msg = (
            f"⚠️ *Confirm Revoke Delegate Authority* ⚠️\n\n"
            f"You are about to revoke any delegate permissions from the following token account:\n"
            f"  Token: *{selected_token_info['symbol']}* ({selected_token_info['name']})\n"
            f"  Account Address: `{token_account_pubkey}`\n\n"
            "This action removes any third-party spending permissions on this specific token account.\n"
            "Do you want to proceed?"
        )
        keyboard = [
            [InlineKeyboardButton("✅ Confirm Revoke", callback_data="confirm_revoke")],
            [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
        return REVOKE_CONFIRMATION

    except ValueError:
        await query.edit_message_text("❌ Invalid token account address format. Please try again or /cancel.")
        context.user_data.clear()
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Error selecting token account for revoke: {e}", exc_info=True)
        await query.edit_message_text("❌ An unexpected error occurred. Please try again or /cancel.")
        context.user_data.clear()
        return ConversationHandler.END

async def execute_revoke_permission(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Executes the revoke token delegate transaction."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    token_account_str = context.user_data.get('revoke_target_token_account')
    if not token_account_str:
        await query.edit_message_text("❌ No token account selected for revoking. Please restart the process.")
        context.user_data.clear()
        return ConversationHandler.END

    wallet = _get_current_wallet(user_id)
    if not wallet:
        await query.edit_message_text("❌ No wallet connected. Cannot execute revoke.")
        context.user_data.clear()
        return ConversationHandler.END

    token_account_pubkey = Pubkey.from_string(token_account_str)

    await query.edit_message_text(f"Attempting to revoke delegate authority from `{token_account_pubkey}`...")

    try:
        tx_signature = await _revoke_token_delegate(SOLANA_RPC_URL, wallet, token_account_pubkey)
        await query.edit_message_text(
            f"✅ Delegate authority successfully revoked!\n"
            f"Account: `{token_account_pubkey}`\n"
            f"Transaction ID: [`{tx_signature}`](https://solscan.io/tx/{tx_signature})"
        , parse_mode="Markdown")
    except RPCException as rpc_e:
        logger.error(f"RPC error revoking permission: {rpc_e}", exc_info=True)
        if "No delegate set" in str(rpc_e):
            await query.edit_message_text(f"ℹ️ No delegate authority was set on `{token_account_pubkey}`. No action needed.")
        else:
            await query.edit_message_text(f"❌ Solana network error during revoke: {rpc_e}.\nPlease try again later.")
    except Exception as e:
        logger.error(f"Error revoking permission: {e}", exc_info=True)
        await query.edit_message_text(f"❌ An unexpected error occurred during revoke: {e}.\nPlease try again or /cancel.")
    finally:
        context.user_data.clear()
        return ConversationHandler.END


# --- General Callbacks (for menu navigation) ---
async def back_to_start_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Returns to the main /start menu."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    msg, reply_markup = await get_start_message(user_id)
    await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")
    context.user_data.clear() # Clear any residual conversation data

async def close_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Closes the current menu message."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Menu closed. Type /start to open it again.")
    context.user_data.clear()

async def generic_under_construction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Placeholder for under-construction features."""
    query = update.callback_query
    await query.answer("This feature is under construction! Please check back later.")
    await query.edit_message_text("🚧 This feature is currently under construction. Please check back later! 🚧",
                                  reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")]])
                                  )

async def market_maker_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    msg = (
        "⚙️ *Market Maker Bot Settings*\n\n"
        "This bot can help you automate trades and provide liquidity to a market.\n"
        "Features include:\n"
        "- Auto-buy/sell based on price movements\n"
        "- Spread management\n"
        "- Liquidity provision strategies\n\n"
        "Please note: This is an advanced feature and requires careful configuration.\n"
        "Currently under development. Stay tuned for updates!"
    )
    keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")

async def tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    msg = (
        "🛠 *STBOT Tools Menu*\n\n"
        "Explore various utilities to enhance your Solana experience:\n"
        "- *Create Token:* Deploy your own SPL token.\n"
        "- *Revoke Permissions:* Manage token account delegate authorities.\n"
        "- *Wallet Generator:* Generate new Solana wallets securely (offline recommendation).\n"
        "- *Transaction Builder:* Advanced tool for custom transaction creation (Coming Soon).\n"
        "- *Devnet Airdrop:* Get test SOL for development on Devnet."
    )
    keyboard = [
        [InlineKeyboardButton("➕ Create New Token", callback_data="create_token")],
        [InlineKeyboardButton("🚫 Revoke Token Permissions", callback_data="revoke_permissions")],
        [InlineKeyboardButton("🔑 Generate New Wallet", callback_data="generate_new_wallet")],
        [InlineKeyboardButton("💧 Get Devnet Airdrop (Test SOL)", callback_data="devnet_airdrop")],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")

async def generate_new_wallet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generates a new Solana wallet and imports it."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    try:
        new_keypair = Keypair()
        new_mnemonic = Mnemonic("english").to_mnemonic(new_keypair.seed)

        if _add_wallet_to_user(user_id, new_keypair):
            await query.edit_message_text(
                f"🎉 *New Wallet Generated & Imported!* 🎉\n\n"
                f"Address: `{new_keypair.pubkey()}`\n"
                f"**IMPORTANT: Save your secret recovery phrase and private key securely OFFLINE!**\n"
                f"Phrase: `{new_mnemonic}`\n"
                f"Private Key (Base58): `{base58.b58encode(new_keypair.secret_key).decode('utf-8')}`\n\n"
                "This is the only time you will see these. If you lose them, you lose access to your funds.\n"
                "Type /wallets_menu to see your wallets or /start to go back to main menu."
            , parse_mode="Markdown")
        else:
            await query.edit_message_text("❌ Failed to add the newly generated wallet. It might already exist.")
    except Exception as e:
        logger.error(f"Error generating new wallet: {e}")
        await query.edit_message_text("❌ An error occurred while generating a new wallet. Please try again.")

    # No specific state needed, just a one-off action.

async def devnet_airdrop_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Requests a SOL airdrop on Devnet for the current wallet."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    wallet = _get_current_wallet(user_id)
    if not wallet:
        await query.edit_message_text("❌ No wallet connected. Please connect a wallet first to request airdrop.")
        return

    await query.edit_message_text(f"Requesting 1 SOL airdrop on Devnet for `{wallet.pubkey()}`... This may take a moment.", parse_mode="Markdown")
    try:
        tx_signature = await _request_devnet_airdrop(str(wallet.pubkey()))
        if tx_signature:
            await query.edit_message_text(
                f"✅ Devnet airdrop successful! 🎉\n"
                f"Transaction ID: [`{tx_signature}`](https://solscan.io/tx/{tx_signature}?cluster=devnet)\n"
                "Note: Airdrops only work on Devnet. Funds are not real SOL."
            , parse_mode="Markdown")
        else:
            await query.edit_message_text("❌ Failed to get Devnet airdrop. Please try again later. Devnet faucet might be rate-limited or empty.")
    except Exception as e:
        logger.error(f"Error during devnet airdrop: {e}", exc_info=True)
        await query.edit_message_text(f"❌ An error occurred during the airdrop: {e}.\nPlease try again.")

# Main function to run the bot
def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Conversation handler for wallet import
    wallet_import_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("importwallet", import_wallet_start), CallbackQueryHandler(add_wallet_prompt, pattern="^add_wallet$")],
        states={
            WALLET_IMPORT: [MessageHandler(filters.TEXT & ~filters.COMMAND, wallet_import_receive)],
        },
        fallbacks=[CommandHandler("cancel", cancel), CallbackQueryHandler(cancel, pattern="^cancel$")]
    )
    application.add_handler(wallet_import_conv_handler)

    # Conversation handler for Buy/Sell
    buy_sell_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_buy_token, pattern="^start_buy_token$"), CallbackQueryHandler(start_sell_token, pattern="^start_sell_token$"),
                      CallbackQueryHandler(lambda update, context: handle_buy_sell_token_address(update, context), pattern="^buy_detected_token_.*"),
                      CallbackQueryHandler(lambda update, context: handle_buy_sell_token_address(update, context), pattern="^sell_detected_token_.*")
                     ],
        states={
            BUY_TOKEN_ADDRESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_buy_sell_token_address)],
            BUY_TOKEN_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_buy_token_amount)],
            SELL_TOKEN_ADDRESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_buy_sell_token_address)],
            SELL_TOKEN_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_sell_token_amount)],
            SWAP_CONFIRMATION: [CallbackQueryHandler(confirm_swap, pattern="^confirm_buy_swap$|^confirm_sell_swap$")],
        },
        fallbacks=[CommandHandler("cancel", cancel), CallbackQueryHandler(cancel, pattern="^cancel$")]
    )
    application.add_handler(buy_sell_conv_handler)

    # Conversation handler for Create Token
    create_token_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(create_token_menu, pattern="^create_token$")],
        states={
            CREATE_TOKEN_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_create_token_name)],
            CREATE_TOKEN_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_create_token_symbol)],
            CREATE_TOKEN_DECIMALS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_create_token_decimals)],
            CREATE_TOKEN_SUPPLY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_create_token_supply)],
        },
        fallbacks=[CommandHandler("cancel", cancel), CallbackQueryHandler(cancel, pattern="^cancel$")]
    )
    application.add_handler(create_token_conv_handler)

    # Conversation handler for Transfer SOL
    transfer_sol_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(transfer_sol_start, pattern="^transfer_sol$")],
        states={
            TRANSFER_SOL_RECIPIENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_transfer_sol_recipient)],
            TRANSFER_SOL_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_transfer_sol_amount)],
        },
        fallbacks=[CommandHandler("cancel", cancel), CallbackQueryHandler(cancel, pattern="^cancel$")]
    )
    application.add_handler(transfer_sol_conv_handler)

    # Conversation handler for Revoke Permissions
    revoke_permissions_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(revoke_permissions_start, pattern="^revoke_permissions$")],
        states={
            REVOKE_TOKEN_MINT: [CallbackQueryHandler(handle_revoke_token_account_selection, pattern="^select_revoke_token_account_.*")],
            REVOKE_CONFIRMATION: [CallbackQueryHandler(execute_revoke_permission, pattern="^confirm_revoke$")],
        },
        fallbacks=[CommandHandler("cancel", cancel), CallbackQueryHandler(cancel, pattern="^cancel$")]
    )
    application.add_handler(revoke_permissions_conv_handler)

    # General Command Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("wallets", wallets_command))
    application.add_handler(CommandHandler("cancel", cancel)) # Global cancel

    # Callback Query Handlers (for menu buttons)
    application.add_handler(CallbackQueryHandler(back_to_start_menu, pattern="^start_menu$"))
    application.add_handler(CallbackQueryHandler(close_menu, pattern="^close$"))
    application.add_handler(CallbackQueryHandler(wallets_menu, pattern="^wallets_menu$"))
    application.add_handler(CallbackQueryHandler(switch_wallet, pattern="^switch_wallet_.*$"))
    application.add_handler(CallbackQueryHandler(remove_current_wallet, pattern="^remove_current_wallet$"))
    application.add_handler(CallbackQueryHandler(export_private_key, pattern="^export_privkey$"))
    application.add_handler(CallbackQueryHandler(show_token_balances, pattern="^show_token_balances$"))
    application.add_handler(CallbackQueryHandler(buy_sell_menu, pattern="^buy_sell$"))
    application.add_handler(CallbackQueryHandler(tools_menu, pattern="^tools$"))
    application.add_handler(CallbackQueryHandler(generate_new_wallet, pattern="^generate_new_wallet$"))
    application.add_handler(CallbackQueryHandler(devnet_airdrop_request, pattern="^devnet_airdrop$"))
    application.add_handler(CallbackQueryHandler(market_maker_menu, pattern="^market_maker$")) # Specific placeholder

    # Under construction handlers for other menu items
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^sniper$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^pumpfun$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^moonshot$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^limit_orders$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^profile$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^trades$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^copy$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^referral$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^settings$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^backup$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^security$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^help$"))
    application.add_handler(CallbackQueryHandler(generic_under_construction, pattern="^tutorials$"))


    # Message handler for token info detection (should be after command handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, token_info_detector))

    # Run the bot
    print("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
