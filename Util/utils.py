"""
@author: Fqf
@time: 20240717
@file: utils.py
@description: Shared Libraries for All users
"""

# ==========================================================
# ======================== Message =========================
# ==========================================================

"""Set green and bold text"""
def HighLightGreenMsg(message):
    highlighted_message = f"\033[1;32m{message}\033[0m"
    return highlighted_message

"""Set red and bold text"""
def HighLightRedMsg(message):
    highlighted_message = f"\033[1;31m{message}\033[0m"
    return highlighted_message

# ==========================================================
# ===================== Visualization ======================
# ==========================================================