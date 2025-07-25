import logging
import os
import re
import time
from datetime import datetime, timedelta
import json

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

from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solana.transaction import Transaction
from solana.system_program import transfer
from solana.rpc.types import TxOpts, TokenAccountOpts
from solana.rpc.api import RPCException

from mnemonic import Mnemonic
import base58
import requests
from dotenv import load_dotenv

from spl.token.async_client import AsyncToken
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import (
    get_associated_token_address,
    revoke,
    RevokeParams
)


# --- Load environment variables ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

# --- Global state ---
# In-memory storage for user wallets.
# For production, replace with persistent database (e.g., Firestore, PostgreSQL)
# Structure: {user_id: {'wallets': [Keypair, ...], 'current_index': 0}}
_user_wallets = {}

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
REVOKE_TOKEN_OWNER = 13

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
        price = await _get_sol_price_from_api()
        if price is not None:
            SOL_PRICE_CACHE["price"] = price
            SOL_PRICE_CACHE["timestamp"] = current_time
        return price

# --- Helper functions (consolidated from modules) ---

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
        if len(secret) == 64: # Full 64-byte secret key (private key + public key)
            return Keypair.from_secret_key(secret)
        elif len(secret) == 32: # Only 32-byte private key seed
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
    """Retrieves the wallet data for a specific user. Initializes if user has no wallets yet."""
    if user_id not in _user_wallets:
        _user_wallets[user_id] = {'wallets': [], 'current_index': 0}
    return _user_wallets[user_id]

def _add_wallet_to_user(user_id: int, keypair: Keypair):
    """Adds a new wallet to a user's list of wallets. Sets it as the current wallet if it's the first one added."""
    user_data = _get_user_wallets(user_id)
    # Check if wallet already exists to prevent duplicates
    if any(kp.public_key == keypair.public_key for kp in user_data['wallets']):
        logger.info(f"Wallet {keypair.public_key} already exists for user {user_id}.")
        return False
    user_data['wallets'].append(keypair)
    if len(user_data['wallets']) == 1: # If it's the first wallet, set it as current
        user_data['current_index'] = 0
    logger.info(f"Wallet {keypair.public_key} added for user {user_id}.")
    return True

def _remove_wallet_from_user(user_id: int, index: int) -> bool:
    """Removes a wallet at a specific index for a user. Adjusts current_index if necessary."""
    user_data = _get_user_wallets(user_id)
    if 0 <= index < len(user_data['wallets']):
        removed_wallet_pubkey = user_data['wallets'][index].public_key
        del user_data['wallets'][index]
        logger.info(f"Wallet {removed_wallet_pubkey} removed for user {user_id}.")
        if not user_data['wallets']: # No wallets left
            user_data['current_index'] = 0
        elif user_data['current_index'] >= len(user_data['wallets']): # Current index now out of bounds
            user_data['current_index'] = len(user_data['wallets']) - 1 # Set to last available wallet
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
            res = await client.get_balance(PublicKey(pubkey_str))
            lamports = res.value
            return lamports / 1e9
    except RPCException as e:
        logger.error(f"Solana RPC error fetching SOL balance for {pubkey_str}: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching SOL balance for {pubkey_str}: {e}")
        return 0.0

async def _get_sol_price_from_api() -> float | None:
    """Fetches the current SOL price in USD from Jupiter Aggregator API."""
    try:
        response = requests.get("https://price.jup.ag/v3/price?ids=SOL")
        response.raise_for_status()
        data = response.json()
        price = data.get('data', {}).get('SOL', {}).get('price')
        if price is None:
            logger.warning("SOL price not found in Jupiter API response.")
            return None
        return float(price)
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error fetching SOL price: {e}")
        return None
    except (KeyError, TypeError) as e:
        logger.error(f"JSON parsing error fetching SOL price: {e}, Response: {response.text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching SOL price: {e}")
        return None

async def _request_devnet_airdrop(pubkey_str: str) -> str | None:
    """Requests a SOL airdrop on Devnet for a given public key."""
    devnet_rpc_url = "https://api.devnet.solana.com"
    try:
        async with AsyncClient(devnet_rpc_url) as client:
            res = await client.request_airdrop(PublicKey(pubkey_str), 1_000_000_000)
            signature = res.value
            if signature:
                await client.confirm_transaction(signature, commitment='confirmed')
                logger.info(f"Devnet airdrop confirmed: {signature}")
                return signature
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
    name: str,
    symbol: str
) -> tuple[PublicKey, str]:
    """Creates a new SPL token mint account."""
    async with AsyncClient(rpc_url) as client:
        mint_keypair = Keypair()
        token = await AsyncToken.create_mint(
            conn=client,
            payer=payer,
            mint_authority=payer.public_key,
            freeze_authority=payer.public_key,
            decimals=decimals,
            program_id=TOKEN_PROGRAM_ID,
            skip_confirmation=False,
            opts=TxOpts(skip_preflight=True)
        )
        tx_signature = token.mint
        logger.info(f"Created SPL Token Mint: {token.pubkey} with Tx: {tx_signature}")
        return token.pubkey, tx_signature

async def _mint_spl_tokens(
    rpc_url: str,
    payer: Keypair,
    mint_pubkey: PublicKey,
    amount: int,
    decimals: int
) -> str:
    """Mints new SPL tokens to the payer's associated token account."""
    async with AsyncClient(rpc_url) as client:
        token_client = AsyncToken(client, mint_pubkey, TOKEN_PROGRAM_ID, payer)
        associated_token_address = get_associated_token_address(payer.public_key, mint_pubkey)

        try:
            await client.get_account_info(associated_token_address)
            logger.info(f"Associated Token Account {associated_token_address} already exists for {payer.public_key}.")
        except RPCException:
            logger.info(f"Creating Associated Token Account {associated_token_address} for {payer.public_key}.")
            pass

        tx_signature = await token_client.mint_to(
            dest=associated_token_address,
            owner=payer,
            amount=amount,
            multi_signers=[],
            opts=TxOpts(skip_preflight=True)
        )
        logger.info(f"Minted {amount} tokens to {associated_token_address} with Tx: {tx_signature}")
        return tx_signature

async def _get_token_accounts_and_balances(rpc_url: str, owner_pubkey: PublicKey) -> dict:
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
                account_info = account.account.data.parsed['info']
                mint_address = account_info['mint']
                token_amount = account_info['tokenAmount']
                amount = float(token_amount['uiAmountString'])
                decimals = token_amount['decimals']

                token_name = f"Token {mint_address[:6]}..."
                token_symbol = f"TKN{mint_address[-4:]}"

                token_balances[mint_address] = {
                    'amount': amount,
                    'decimals': decimals,
                    'name': token_name,
                    'symbol': token_symbol
                }
        except RPCException as e:
            logger.error(f"Solana RPC error fetching token accounts for {owner_pubkey}: {e}")
        except Exception as e:
            logger.error(f"Error fetching token accounts for {owner_pubkey}: {e}")
    return token_balances

async def _revoke_token_delegate(
    rpc_url: str,
    owner_wallet: Keypair,
    mint_pubkey: PublicKey,
    token_account_owner_pubkey: PublicKey
) -> str:
    """Revokes any delegate authority on a specific token account."""
    async with AsyncClient(rpc_url) as client:
        token_account_pubkey = get_associated_token_address(token_account_owner_pubkey, mint_pubkey)

        revoke_ix = revoke(RevokeParams(
            program_id=TOKEN_PROGRAM_ID,
            account=token_account_pubkey,
            owner=owner_wallet.public_key,
            signers=[]
        ))
        recent_blockhash = (await client.get_latest_blockhash()).value.blockhash
        transaction = Transaction(recent_blockhash=recent_blockhash)
        transaction.add(revoke_ix)
        try:
            signature_resp = await client.send_transaction(transaction, owner_wallet, opts=TxOpts(skip_preflight=True))
            signature = signature_resp['result']
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
        sol_balance = await _get_sol_balance(SOLANA_RPC_URL, str(wallet.public_key))
        usd_val = sol_balance * sol_price if sol_price is not None else 0

        msg_parts.append(
            "💳 Your Current Wallet:\n"
            f"↳ `{wallet.public_key}` 🅴\n"
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
    """Handles the /start command, showing initial bot message and main menu."""
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    msg, reply_markup = await get_start_message(user_id)
    await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode="Markdown")

async def wallets_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /wallets command, showing current wallet info or prompt to connect."""
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_wallet_index = user_data.get('current_index', 0)

    if wallets and current_wallet_index < len(wallets):
        wallet = wallets[current_wallet_index]
        sol_balance = await _get_sol_balance(SOLANA_RPC_URL, str(wallet.public_key))
        sol_price = await get_cached_sol_price() or 0
        usd_val = sol_balance * sol_price

        msg = (
            "💳 Your Current Wallet:\n"
            f"Address: `{wallet.public_key}` 🅴\n"
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
    current_wallet_index = user_data.get('current_index', 0)

    msg = "💼 *Wallet Management*\n\n"
    if not wallets:
        msg += "You have no wallets connected. Use /importwallet to add one.\n"
    else:
        msg += "Your connected wallets:\n"
        for i, wallet in enumerate(wallets):
            status = "✅ (Current)" if i == current_wallet_index else ""
            msg += f"  `{str(wallet.public_key)[:6]}...{str(wallet.public_key)[-4:]}` {status}\n"
        msg += "\nSelect a wallet to switch, or use the options below:\n"

    keyboard = []
    for i, wallet in enumerate(wallets):
        keyboard.append([InlineKeyboardButton(f"Switch to Wallet {i+1} ({str(wallet.public_key)[:6]}...)", callback_data=f"switch_wallet_{i}")])

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
                                          f"Current wallet: `{wallets[index_to_switch].public_key}`")
            msg, reply_markup = await get_start_message(user_id)
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
        "⚠️ WARNING: Importing wallets by secret phrase or private key is risky.\n"
        "Do not share to public group.\n\n"
        "Send your 12/24-word secret phrase OR your base58 private key string now.\n"
        "Or /cancel to abort."
    )
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
        await wallets_menu(update, context)
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

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets or current_index >= len(wallets):
        await query.edit_message_text("❌ No wallet connected to export.")
        return

    wallet = wallets[current_index]
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

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets or current_index >= len(wallets):
        await query.edit_message_text("❌ No wallet connected to view token balances.")
        return

    wallet = wallets[current_index]
    await query.edit_message_text(f"Fetching token balances for `{wallet.public_key}`...")

    try:
        token_balances = await _get_token_accounts_and_balances(SOLANA_RPC_URL, wallet.public_key)
        if not token_balances:
            msg = "No SPL tokens found in this wallet."
        else:
            msg = "📊 *Your Token Balances:*\n\n"
            for mint_address, details in token_balances.items():
                formatted_amount = f"{details['amount']:.{details['decimals']}f}".rstrip('0').rstrip('.')
                msg += (
                    f"*{details['symbol']}* ({details['name']})\n"
                    f"  Balance: {formatted_amount}\n"
                    f"  Mint: `{mint_address}`\n\n"
                )
        await query.edit_message_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error fetching token balances: {e}")
        await query.edit_message_text("❌ Failed to fetch token balances. Please try again later.")

    keyboard = [[InlineKeyboardButton("⬅️ Back to Wallet Menu", callback_data="wallets_menu")]]
    await query.message.reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))


async def import_wallet_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the conversation for importing a wallet."""
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    await update.message.reply_text(
        "⚠️ WARNING: Do not share secret phrase or private key to public chats.\n"
        " ... .\n\n"
        "Send your 12/24-word secret phrase OR your base58 private key string now.\n"
        "Or /cancel to abort."
    )
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
                f"✅ Wallet imported successfully.\nAddress: `{wallet.public_key}`\n⚠️ Only use test wallets.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                f"ℹ️ Wallet `{wallet.public_key}` was already connected. No new wallet added.",
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
    return ConversationHandler.END

async def token_info_detector(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Detects if a message is a Solana token address and provides a link to explore it."""
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    text = update.message.text.strip()
    if re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", text):
        try:
            PublicKey(text)
            is_valid_solana_address = True
        except Exception:
            is_valid_solana_address = False

        if is_valid_solana_address:
            await update.message.reply_text(
                f"Detected a potential Solana address: `{text}`\n\n"
                "You can explore this address/token on Solana explorers:\n"
                f"[Solscan](https://solscan.io/token/{text})\n"
                f"[Birdeye](https://birdeye.so/token/{text})\n"
                f"[Jupiter Terminal](https://terminal.jup.ag/swap?inputMint=SOL&outputMint={text})"
                "\n\n*(Note: Detailed token info from Birdeye is not integrated in this version.)*",
                parse_mode="Markdown",
                disable_web_page_preview=True
            )

async def airdrop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Airdrops test SOL to the current wallet on devnet."""
    user_id = update.effective_user.id
    if is_rate_limited(user_id):
        await update.message.reply_text("Please wait a moment before sending another command.")
        return

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets or current_index >= len(wallets):
        await update.message.reply_text("❌ No wallet connected. Please import one first.")
        return

    wallet = wallets[current_index]
    await update.message.reply_text(f"Requesting Devnet airdrop for `{wallet.public_key}`...")
    try:
        tx_signature = await _request_devnet_airdrop(str(wallet.public_key))
        if tx_signature:
            await update.message.reply_text(f"✅ Devnet airdrop requested!\nTx Signature: `{tx_signature}`")
        else:
            await update.message.reply_text(f"❌ Airdrop failed to return a signature. It might be a network issue or rate limit.")
    except Exception as e:
        logger.error(f"Airdrop failed: {e}")
        await update.message.reply_text(f"❌ Airdrop failed: {e}. Please try again later.")

# --- Trading Logic Placeholders ---

async def buy_sell_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Displays the buy/sell menu."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    keyboard = [
        [InlineKeyboardButton("💰 Buy Token", callback_data="buy_token")],
        [InlineKeyboardButton("💸 Sell Token", callback_data="sell_token")],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("🚀 *Buy & Sell Tokens*\n\nChoose an action:", reply_markup=reply_markup, parse_mode="Markdown")

async def buy_token_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the buy token conversation."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets or current_index >= len(wallets):
        await query.edit_message_text("❌ No wallet connected. Please import one first.")
        return

    await query.edit_message_text("Please send the *token address* you wish to buy.")
    return BUY_TOKEN_ADDRESS

async def receive_buy_token_address(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the token address for buying."""
    token_address_str = update.message.text.strip()
    try:
        PublicKey(token_address_str)
    except Exception:
        await update.message.reply_text("❌ Invalid token address. Please send a valid Solana token address.")
        return BUY_TOKEN_ADDRESS

    context.user_data['buy_token_address'] = token_address_str
    await update.message.reply_text("Now, please send the *amount of SOL* you want to spend (e.g., `0.1` for 0.1 SOL).")
    return BUY_TOKEN_AMOUNT

async def receive_buy_token_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the SOL amount for buying and initiates the transaction."""
    try:
        sol_amount = float(update.message.text.strip())
        if sol_amount <= 0:
            raise ValueError("Amount must be positive.")
    except ValueError:
        await update.message.reply_text("❌ Invalid SOL amount. Please send a positive number (e.g., `0.1`).")
        return BUY_TOKEN_AMOUNT

    user_id = update.message.from_user.id
    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)
    wallet = wallets[current_index]

    token_address_str = context.user_data.get('buy_token_address')
    if not token_address_str:
        await update.message.reply_text("An error occurred. Please restart the buy process with /buy_sell.")
        return ConversationHandler.END

    await update.message.reply_text(f"Preparing to buy token `{token_address_str}` with {sol_amount} SOL...")

    try:
        async with AsyncClient(SOLANA_RPC_URL) as client:
            current_sol_balance = await _get_sol_balance(SOLANA_RPC_URL, str(wallet.public_key))
            if current_sol_balance < sol_amount:
                await update.message.reply_text(
                    f"❌ Insufficient SOL balance. You have {current_sol_balance:.4f} SOL, but need {sol_amount} SOL."
                )
                return ConversationHandler.END

            await update.message.reply_text(
                f"✅ Simulated buy of `{token_address_str}` with {sol_amount} SOL successful!\n"
                "*(Note: This is a placeholder. Actual token swap logic needs to be implemented using a DEX aggregator like Jupiter.)*"
            )

    except RPCException as e:
        logger.error(f"Solana RPC error during buy: {e}")
        await update.message.reply_text(f"❌ Solana network error during buy: {e.args[0]['message']}")
    except Exception as e:
        logger.error(f"Error during token buy: {e}")
        await update.message.reply_text(f"❌ Failed to execute buy: {e}. Please ensure you have enough SOL and the token address is valid.")

    return ConversationHandler.END

async def sell_token_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the sell token conversation."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets or current_index >= len(wallets):
        await query.edit_message_text("❌ No wallet connected. Please import one first.")
        return

    await query.edit_message_text("Please send the *token address* you wish to sell.")
    return SELL_TOKEN_ADDRESS

async def receive_sell_token_address(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the token address for selling."""
    token_address_str = update.message.text.strip()
    try:
        PublicKey(token_address_str)
    except Exception:
        await update.message.reply_text("❌ Invalid token address. Please send a valid Solana token address.")
        return SELL_TOKEN_ADDRESS

    context.user_data['sell_token_address'] = token_address_str
    await update.message.reply_text("Now, please send the *amount of tokens* you want to sell (e.g., `100`).")
    return SELL_TOKEN_AMOUNT

async def receive_sell_token_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the token amount for selling and initiates the transaction."""
    try:
        token_amount = float(update.message.text.strip())
        if token_amount <= 0:
            raise ValueError("Amount must be positive.")
    except ValueError:
        await update.message.reply_text("❌ Invalid token amount. Please send a positive number (e.g., `100`).")
        return SELL_TOKEN_AMOUNT

    user_id = update.message.from_user.id
    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)
    wallet = wallets[current_index]

    token_address_str = context.user_data.get('sell_token_address')
    if not token_address_str:
        await update.message.reply_text("An error occurred. Please restart the sell process with /buy_sell.")
        return ConversationHandler.END

    await update.message.reply_text(f"Preparing to sell {token_amount} of token `{token_address_str}`...")

    try:
        await update.message.reply_text(
            f"✅ Simulated sell of {token_amount} of `{token_address_str}` successful!\n"
            "*(Note: This is a placeholder. Actual token swap logic needs to be implemented using a DEX aggregator like Jupiter.)*"
        )

    except RPCException as e:
        logger.error(f"Solana RPC error during sell: {e}")
        await update.message.reply_text(f"❌ Solana network error during sell: {e.args[0]['message']}")
    except Exception as e:
        logger.error(f"Error during token sell: {e}")
        await update.message.reply_text(f"❌ Failed to execute sell: {e}. Please ensure you have enough tokens and the address is valid.")

    return ConversationHandler.END

# --- SPL Token Creation Logic ---

async def create_token_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the SPL token creation conversation."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets or current_index >= len(wallets):
        await query.edit_message_text("❌ No wallet connected. Please import one first to create tokens.")
        return

    await query.edit_message_text("Let's create a new SPL token!\n\nPlease send the *name* for your token (e.g., `My Awesome Token`).")
    return CREATE_TOKEN_NAME

async def receive_token_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the token name."""
    token_name = update.message.text.strip()
    if not token_name:
        await update.message.reply_text("❌ Token name cannot be empty. Please send a name.")
        return CREATE_TOKEN_NAME
    context.user_data['new_token_name'] = token_name
    await update.message.reply_text(f"Great! Now send the *symbol* for '{token_name}' (e.g., `MAT`).")
    return CREATE_TOKEN_SYMBOL

async def receive_token_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the token symbol."""
    token_symbol = update.message.text.strip()
    if not token_symbol or len(token_symbol) > 10:
        await update.message.reply_text("❌ Invalid symbol. Please send a short symbol (e.g., `MAT`, max 10 chars).")
        return CREATE_TOKEN_SYMBOL
    context.user_data['new_token_symbol'] = token_symbol
    await update.message.reply_text(f"Okay, symbol is '{token_symbol}'. Now, how many *decimals* should your token have? (e.g., `9` for standard precision).")
    return CREATE_TOKEN_DECIMALS

async def receive_token_decimals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the token decimals."""
    try:
        decimals = int(update.message.text.strip())
        if not (0 <= decimals <= 9):
            raise ValueError("Decimals must be between 0 and 9.")
    except ValueError:
        await update.message.reply_text("❌ Invalid decimals. Please send a number between 0 and 9 (e.g., `9`).")
        return CREATE_TOKEN_DECIMALS
    context.user_data['new_token_decimals'] = decimals
    await update.message.reply_text(f"Decimals set to {decimals}. Finally, what is the *total supply* for your token? (e.g., `1000000` for 1 million).")
    return CREATE_TOKEN_SUPPLY

async def receive_token_supply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the token supply and initiates token creation."""
    try:
        total_supply = int(update.message.text.strip())
        if total_supply <= 0:
            raise ValueError("Supply must be positive.")
    except ValueError:
        await update.message.reply_text("❌ Invalid supply. Please send a positive integer (e.g., `1000000`).")
        return CREATE_TOKEN_SUPPLY

    user_id = update.message.from_user.id
    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)
    payer_wallet = wallets[current_index]

    token_name = context.user_data.get('new_token_name')
    token_symbol = context.user_data.get('new_token_symbol')
    decimals = context.user_data.get('new_token_decimals')

    await update.message.reply_text(f"Creating token '{token_name}' ({token_symbol}) with {decimals} decimals and total supply {total_supply}...")

    try:
        mint_pubkey, tx_sig_create_mint = await _create_spl_token_mint(
            SOLANA_RPC_URL, payer_wallet, decimals, token_name, token_symbol
        )
        await update.message.reply_text(
            f"✅ Token Mint created! Mint Address: `{mint_pubkey}`\n"
            f"Tx Signature: `{tx_sig_create_mint}`"
        )

        raw_supply_amount = total_supply * (10 ** decimals)
        tx_sig_mint_supply = await _mint_spl_tokens(
            SOLANA_RPC_URL, payer_wallet, mint_pubkey, raw_supply_amount, decimals
        )
        await update.message.reply_text(
            f"✅ Initial supply minted to your wallet!\n"
            f"Tx Signature: `{tx_sig_mint_supply}`\n\n"
            f"Your new token '{token_name}' is ready!"
        )

    except RPCException as e:
        logger.error(f"Solana RPC error during token creation: {e}")
        await update.message.reply_text(f"❌ Solana network error during token creation: {e.args[0]['message']}")
    except Exception as e:
        logger.error(f"Error creating SPL token: {e}")
        await update.message.reply_text(f"❌ Failed to create SPL token: {e}. Ensure you have enough SOL for transaction fees.")

    return ConversationHandler.END

# --- Transfer SOL Logic ---
async def transfer_sol_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the SOL transfer conversation."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets or current_index >= len(wallets):
        await query.edit_message_text("❌ No wallet connected. Please import one first to transfer SOL.")
        return

    await query.edit_message_text("Please send the *recipient's Solana address*.")
    return TRANSFER_SOL_RECIPIENT

async def receive_transfer_recipient(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the recipient address for SOL transfer."""
    recipient_address_str = update.message.text.strip()
    try:
        PublicKey(recipient_address_str)
    except Exception:
        await update.message.reply_text("❌ Invalid recipient address. Please send a valid Solana public key.")
        return TRANSFER_SOL_RECIPIENT

    context.user_data['transfer_recipient'] = recipient_address_str
    await update.message.reply_text(f"Recipient set to `{recipient_address_str}`. Now, how much *SOL* do you want to send? (e.g., `0.05`).")
    return TRANSFER_SOL_AMOUNT

async def receive_transfer_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the SOL amount for transfer and initiates the transaction."""
    try:
        sol_amount = float(update.message.text.strip())
        if sol_amount <= 0:
            raise ValueError("Amount must be positive.")
    except ValueError:
        await update.message.reply_text("❌ Invalid SOL amount. Please send a positive number (e.g., `0.05`).")
        return TRANSFER_SOL_AMOUNT

    user_id = update.message.from_user.id
    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)
    sender_wallet = wallets[current_index]

    recipient_address_str = context.user_data.get('transfer_recipient')
    if not recipient_address_str:
        await update.message.reply_text("An error occurred. Please restart the transfer process.")
        return ConversationHandler.END

    await update.message.reply_text(f"Attempting to transfer {sol_amount} SOL to `{recipient_address_str}`...")

    try:
        async with AsyncClient(SOLANA_RPC_URL) as client:
            current_sol_balance = await _get_sol_balance(SOLANA_RPC_URL, str(sender_wallet.public_key))
            estimated_fee_sol = 0.00001
            if current_sol_balance < (sol_amount + estimated_fee_sol):
                await update.message.reply_text(
                    f"❌ Insufficient SOL balance. You have {current_sol_balance:.4f} SOL, "
                    f"but need approximately {sol_amount + estimated_fee_sol:.4f} SOL (including fees)."
                )
                return ConversationHandler.END

            lamports = int(sol_amount * 1e9)
            recent_blockhash = (await client.get_latest_blockhash()).value.blockhash

            transaction = Transaction(recent_blockhash=recent_blockhash)
            transaction.add(transfer(
                from_pubkey=sender_wallet.public_key,
                to_pubkey=PublicKey(recipient_address_str),
                lamports=lamports
            ))

            signature_resp = await client.send_transaction(transaction, sender_wallet, opts=TxOpts(skip_preflight=True))
            signature = signature_resp['result']
            await update.message.reply_text(
                f"✅ SOL transfer successful!\n"
                f"Tx Signature: `{signature}`\n"
                "Please check a block explorer for confirmation."
            )

    except RPCException as e:
        logger.error(f"Solana RPC error during SOL transfer: {e}")
        await update.message.reply_text(f"❌ Solana network error during transfer: {e.args[0]['message']}. Please check your SOL balance and try again.")
    except Exception as e:
        logger.error(f"Error during SOL transfer: {e}")
        await update.message.reply_text(f"❌ Failed to transfer SOL: {e}. Please ensure the recipient address is correct and try again.")

    return ConversationHandler.END

# --- Revoke Token Permissions Logic ---
async def revoke_permissions_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the revoke token permissions conversation."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)

    if not wallets or current_index >= len(wallets):
        await query.edit_message_text("❌ No wallet connected. Please import one first to revoke permissions.")
        return

    await query.edit_message_text(
        "To revoke token permissions, I need the *mint address* of the token.\n"
        "This will revoke any 'delegate' permissions you might have granted for this specific token."
    )
    return REVOKE_TOKEN_MINT

async def receive_revoke_token_mint(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the token mint address for revoking permissions."""
    token_mint_str = update.message.text.strip()
    try:
        PublicKey(token_mint_str)
    except Exception:
        await update.message.reply_text("❌ Invalid token mint address. Please send a valid Solana public key.")
        return REVOKE_TOKEN_MINT

    context.user_data['revoke_token_mint'] = token_mint_str
    await update.message.reply_text(
        f"Token mint set to `{token_mint_str}`. Now, please confirm the *owner's public key* (your wallet address) "
        "for this token account. This is usually your own wallet address."
    )
    return REVOKE_TOKEN_OWNER

async def receive_revoke_token_owner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives the owner's public key and attempts to revoke token permissions."""
    owner_pubkey_str = update.message.text.strip()
    try:
        PublicKey(owner_pubkey_str)
    except Exception:
        await update.message.reply_text("❌ Invalid owner public key. Please send a valid Solana public key.")
        return REVOKE_TOKEN_OWNER

    user_id = update.message.from_user.id
    user_data = _get_user_wallets(user_id)
    wallets = user_data.get('wallets', [])
    current_index = user_data.get('current_index', 0)
    payer_wallet = wallets[current_index]

    token_mint_str = context.user_data.get('revoke_token_mint')
    if not token_mint_str:
        await update.message.reply_text("An error occurred. Please restart the revoke process.")
        return ConversationHandler.END

    if str(payer_wallet.public_key) != owner_pubkey_str:
        await update.message.reply_text(
            "❌ The provided owner public key does not match your currently connected wallet.\n"
            "Please ensure you are trying to revoke permissions for a token account owned by your active wallet."
        )
        return REVOKE_TOKEN_OWNER

    await update.message.reply_text(
        f"Attempting to revoke permissions for token mint `{token_mint_str}` "
        f"from owner `{owner_pubkey_str}`..."
    )

    try:
        tx_signature = await _revoke_token_delegate(
            SOLANA_RPC_URL, payer_wallet, PublicKey(token_mint_str), PublicKey(owner_pubkey_str)
        )
        await update.message.reply_text(
            f"✅ Token permissions revoked successfully!\n"
            f"Tx Signature: `{tx_signature}`\n"
            "This revokes any 'delegate' authority on your token account for this mint."
        )
    except RPCException as e:
        logger.error(f"Solana RPC error during revoke permissions: {e}")
        await update.message.reply_text(f"❌ Solana network error: {e.args[0]['message']}. Ensure the token account exists and has a delegate to revoke.")
    except Exception as e:
        logger.error(f"Error revoking permissions: {e}")
        await update.message.reply_text(f"❌ Failed to revoke permissions: {e}. Please try again.")

    return ConversationHandler.END

# --- Callback query handlers for main menu buttons ---
async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles all inline keyboard button presses."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    if is_rate_limited(user_id):
        await query.message.reply_text("Please wait a moment before sending another command.")
        return

    data = query.data

    menu_actions = {
        "start_menu": start_menu_action,
        "buy_sell": buy_sell_menu,
        "wallets_menu": wallets_menu,
        "add_wallet": add_wallet_prompt,
        "remove_current_wallet": remove_current_wallet,
        "export_privkey": export_private_key,
        "show_token_balances": show_token_balances,
        "tools": tools_menu_action,
        "security": security_message_action,
        "help": help_message_action,
        "tutorials": tutorials_message_action,
        "close": close_message_action,
        "sniper": generic_coming_soon,
        "pumpfun": generic_coming_soon,
        "moonshot": generic_coming_soon,
        "limit_orders": generic_coming_soon,
        "profile": generic_coming_soon,
        "trades": generic_coming_soon,
        "copy": generic_coming_soon,
        "referral": generic_coming_soon,
        "settings": generic_coming_soon,
        "market_maker": generic_coming_soon,
        "backup": generic_coming_soon,
    }

    if data.startswith("switch_wallet_"):
        await switch_wallet(update, context)
    elif data in menu_actions:
        if data in ["security", "help", "tutorials", "close"]:
            await menu_actions[data](update, context, data)
        elif data in ["sniper", "pumpfun", "moonshot", "limit_orders", "profile", "trades", "copy", "referral", "settings", "market_maker", "backup"]:
            await menu_actions[data](update, context)
        else:
            await menu_actions[data](update, context)
    elif data == "buy_token":
        return await buy_token_start(update, context)
    elif data == "sell_token":
        return await sell_token_start(update, context)
    elif data == "create_spl_token":
        return await create_token_start(update, context)
    elif data == "transfer_sol":
        return await transfer_sol_start(update, context)
    elif data == "revoke_permissions":
        return await revoke_permissions_start(update, context)


async def start_menu_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Helper for 'start_menu' callback."""
    msg, reply_markup = await get_start_message(update.callback_query.from_user.id)
    await update.callback_query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")

async def tools_menu_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Helper for 'tools' menu callback."""
    keyboard = [
        [InlineKeyboardButton("➕ Create SPL Token", callback_data="create_spl_token")],
        [InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text("🔥 *Our STBOT Tools*\n\nChoose a tool:", reply_markup=reply_markup, parse_mode="Markdown")

async def generic_coming_soon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generic handler for features coming soon."""
    button_text = next((b.text for row in update.callback_query.message.reply_markup.inline_keyboard for b in row if b.callback_data == update.callback_query.data), "This feature")
    await update.callback_query.edit_message_text(f"{button_text} functionality coming soon!")
    keyboard = [[InlineKeyboardButton("⬅️ Back to Main Menu", callback_data="start_menu")]]
    await update.callback_query.message.reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))

async def security_message_action(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str):
    await update.callback_query.edit_message_text("🛡 Security tips: Always use test wallets for new bots! Never share your private keys or seed phrases.\n\n"
                                                  "⬅️ Back to Main Menu", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Back", callback_data="start_menu")]]))

async def help_message_action(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str):
    await update.callback_query.edit_message_text("ℹ️ Help: For assistance, contact support@soltradingbot.xyz or visit our website.\n\n"
                                                  "⬅️ Back to Main Menu", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Back", callback_data="start_menu")]]))

async def tutorials_message_action(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str):
    await update.callback_query.edit_message_text("📄 Tutorials: Find guides on our website and YouTube channel.\n\n"
                                                  "⬅️ Back to Main Menu", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Back", callback_data="start_menu")]]))

async def close_message_action(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str):
    await update.callback_query.edit_message_text("Bot closed. Type /start to restart.")


# --- Main function ---
def main():
    """Main function to run the Telegram bot."""
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not found in .env file. Please set it.")
        return

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    import_wallet_conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('importwallet', import_wallet_start),
            CallbackQueryHandler(add_wallet_prompt, pattern='^add_wallet$')
        ],
        states={
            WALLET_IMPORT: [MessageHandler(filters.TEXT & ~filters.COMMAND, wallet_import_receive)]
        },
        fallbacks=[CommandHandler('cancel', cancel), CallbackQueryHandler(cancel, pattern='^cancel$')],
    )
    application.add_handler(import_wallet_conv_handler)

    buy_token_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(buy_token_start, pattern='^buy_token$')],
        states={
            BUY_TOKEN_ADDRESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_buy_token_address)],
            BUY_TOKEN_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_buy_token_amount)],
        },
        fallbacks=[CommandHandler('cancel', cancel), CallbackQueryHandler(cancel, pattern='^cancel$')],
    )
    application.add_handler(buy_token_conv_handler)

    sell_token_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(sell_token_start, pattern='^sell_token$')],
        states={
            SELL_TOKEN_ADDRESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_sell_token_address)],
            SELL_TOKEN_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_sell_token_amount)],
        },
        fallbacks=[CommandHandler('cancel', cancel), CallbackQueryHandler(cancel, pattern='^cancel$')],
    )
    application.add_handler(sell_token_conv_handler)

    create_token_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(create_token_start, pattern='^create_spl_token$')],
        states={
            CREATE_TOKEN_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_token_name)],
            CREATE_TOKEN_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_token_symbol)],
            CREATE_TOKEN_DECIMALS: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_token_decimals)],
            CREATE_TOKEN_SUPPLY: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_token_supply)],
        },
        fallbacks=[CommandHandler('cancel', cancel), CallbackQueryHandler(cancel, pattern='^cancel$')],
    )
    application.add_handler(create_token_conv_handler)

    transfer_sol_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(transfer_sol_start, pattern='^transfer_sol$')],
        states={
            TRANSFER_SOL_RECIPIENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_transfer_recipient)],
            TRANSFER_SOL_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_transfer_amount)],
        },
        fallbacks=[CommandHandler('cancel', cancel), CallbackQueryHandler(cancel, pattern='^cancel$')],
    )
    application.add_handler(transfer_sol_conv_handler)

    revoke_permissions_conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(revoke_permissions_start, pattern='^revoke_permissions$')],
        states={
            REVOKE_TOKEN_MINT: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_revoke_token_mint)],
            REVOKE_TOKEN_OWNER: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_revoke_token_owner)],
        },
        fallbacks=[CommandHandler('cancel', cancel), CallbackQueryHandler(cancel, pattern='^cancel$')],
    )
    application.add_handler(revoke_permissions_conv_handler)


    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('wallets', wallets_command))
    application.add_handler(CommandHandler('airdrop', airdrop_command))
    application.add_handler(CommandHandler('cancel', cancel))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, token_info_detector))

    application.add_handler(CallbackQueryHandler(handle_callback_query))

    logger.info("Bot started and polling...")
    application.run_polling()

if __name__ == "__main__":
    main()
