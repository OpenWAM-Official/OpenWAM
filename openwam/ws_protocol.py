"""WebSocket message protocol for the OpenWAM policy server.

Single source of truth for the message-type vocabulary shared by the server
(``openwam/deploy/policy_server.py``) and the clients
(``benchmarks/utils/transport.py``, ``benchmarks/robotwin/openwam2robotwin_interface.py``).

Kept at the torch-free package top level on purpose: the thin RoboTwin client
env (numpy / opencv / Pillow, no torch) imports it, so it must not pull in the
heavy ``openwam.deploy`` package.
"""

# --- Message types ---
# Client → server
OBS = "obs"
RESET = "reset"
PING = "ping"
# Server → client
ACTION = "action"
RESET_ACK = "reset_ack"
PONG = "pong"
ERROR = "error"

# --- Error codes (the "code" field of an ERROR message) ---
ERR_UNKNOWN_TYPE = "unknown_message_type"
ERR_OBS_VALIDATION = "obs_validation_error"
ERR_INTERNAL = "internal_error"
