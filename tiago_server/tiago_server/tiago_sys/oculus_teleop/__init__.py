INTERFACE_MAP = {}

def try_import(module, class_name, interface_name):
    try:
        interface_class = __import__(module, fromlist=[class_name])
        INTERFACE_MAP[interface_name] = getattr(interface_class, class_name)
    except ImportError as e:
        print(e)

try_import('tiago_onboard.tiago_sys.oculus_teleop.oculus', 'OculusPolicy', 'oculus')