from params_proto.v2.proto import ParamsProto, get_children


class Root(ParamsProto, prefix="root"):
  launch_type = "borg"

  # def __init__(self, _deps=None, **_):
  #     super().__init__(**_)


if __name__ == "__main__":
  assert vars(Root) == {"launch_type": "borg"}

  r = Root()
  assert r.launch_type == "borg", f"`{r.launch_type}` should be `borg`"

  r = Root({"root.launch_type": "local"})
  assert r.launch_type == "local", f"`{r.launch_type}` should be `local`"


class SomeConfig(ParamsProto, prefix="resource"):
  fixed = "default"
  some_item = "default_value"

  @get_children
  def __init__(self, _deps=None, **children):
    root = Root(_deps)  # this updates the root object.
    if root.launch_type == "borg":
      self.some_item = "new_value"
    else:
      self.some_item = self.__class__.some_item


if __name__ == "__main__":
  s = SomeConfig({"root.launch_type": "local"})
  assert s.some_item == "default_value"

  s = SomeConfig({"root.launch_type": "borg"})
  assert s.some_item == "new_value"


class Teacher(ParamsProto, prefix="resources.teacher"):
  cell = None
  autopilot = False

  @get_children
  def __init__(self, _deps, **children):
    # if Root(_deps).launch_type != 'local':
    #     self.replicas_hint = children.get('replicas_hint', 26)
    super().__init__(**children)


class Resources(ParamsProto):
  @get_children
  def __init__(self, _deps=None, **children):
    r = Root(_deps)
    print(children)
    self.item = children.get("teacher", None)
    self.teacher = Teacher(_deps, replicas_hint=26 if r.launch_type == "borg" else 1)
    self.bad_teacher = Teacher(
      _deps, _prefix="bad_teacher", replicas_hint=26 if r.launch_type == "borg" else 1
    )


if __name__ == "__main__":
  sweep_param = {
    "root.launch_type": "local",
    "resources.teacher.replica_hint": 10,
  }

  gd = Resources(sweep_param)
  print(vars(gd))
  print(gd.teacher)
  print(gd.bad_teacher)
