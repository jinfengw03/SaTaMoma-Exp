from tiago_client.utils.general_utils import AttrDict

teleop_config = AttrDict(
    arm_left_controller=None,
    arm_right_controller=None,
    base_controller=None,
    torso_controller=None,
    use_oculus=False,
    interface_kwargs=AttrDict(
        oculus={
            # Set these if you want network mode; leave None to use USB like reader.py
            'ip_address': None,
            'port': 5555,
        },
        vision={},
        mobile_phone={},
        spacemouse={},
        keyboard={},
    )
)