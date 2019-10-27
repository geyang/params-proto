from params_proto.neo_proto import ParamsProto

class Root(ParamsProto):
    launch_type = "borg"


class Config(ParamsProto):
    """
    Your ICLR best paper project

    -- Ge
    """
    seed = 10

    def __init__(self, _dep=None):
        r = Root(_dep)

        if r.launch_type == "borg":
            self.replica_hint = 26
        else:
            self.replica_hint = 1


if __name__ == "__main__":



    print(f"seed is {Config().seed}")
    print(f"seed is {Config().replica_hint}")

    for launch_type in ['borg', 'local']:
        config = Config({"Root.launch_type": launch_type})
        assert config.replica_hint == 26 if launch_type == "borg" else 1
        print(config.replica_hint)
