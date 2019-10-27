class Root:
    some_mode = True


class Dependent:
    fixed = "default"
    some_item = "default value"

    def __init__(self):
        root = Root()  # this updates the root object.
        self.some_item = self.__class__.some_item if root.some_mode else "new_value"


class GrandDependent:
    def __init__(self):
        d = Dependent()  # this returns an instance of the dependent
        if d.some_item == "new_value":
            self.item = "hey"
        else:
            self.item = "yo"


def train_fn():  # _Root:Root, _Dependent:Dependent
    # done: add DAT resolution.
    Root.some_mode = True
    d = Dependent()
    assert d.some_item == "default value"

    Root.some_mode = False
    d = Dependent()
    assert Dependent.some_item == "default value", "the original remain unchanged."
    assert d.fixed == "default", "this should be 'default'"
    assert d.some_item == "new_value", "this should be 'new value'"

    Root.some_mode = True
    gd = GrandDependent()
    assert gd.item == "yo"

    Root.some_mode = False
    gd = GrandDependent()
    assert gd.item == "hey"


if __name__ == "__main__":
    train_fn()
    from termcolor import cprint

    cprint("all tests have passed!!", "green")
