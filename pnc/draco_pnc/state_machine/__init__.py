class LocomanipulationState(object):
    STAND = 0
    BALANCE = 1
    RF_CONTACT_TRANS_START = 2
    RF_CONTACT_TRANS_END = 3
    RF_SWING = 4
    LF_CONTACT_TRANS_START = 5
    LF_CONTACT_TRANS_END = 6
    LF_SWING = 7

    DH_MANIPULATION = 14


from .contact_transition_end import ContactTransitionEnd
from .contact_transition_start import ContactTransitionStart
from .double_support_balance import DoubleSupportBalance
from .double_support_stand import DoubleSupportStand
from .manipulation import Manipulation
from .single_support_swing import SingleSupportSwing
