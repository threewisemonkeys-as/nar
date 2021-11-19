class Primitive:
    def __init__(self, name, f):
        self._name = name
        self._f = f

    def apply(self, elem):
        return self._f(elem)

    def __str__(self):
        return self.name

    @property
    def name(self):
        return self._name

class AffineTransform(Primitive):
    def __init__(self,name,x,y):
        self._name=name
        self.x=x
        self.y=y
    
    def apply(self,elem):
        tokens=re.findall(r'0|90|180|270|-?[0-9]|[a-z]+',elem[0])
        if(len(tokens)==2):
            return (elem[0]+str(self.x)+str(self.y),elem[1])
        else:
            ans=tokens[0]+tokens[1]
            x=int(tokens[2])+self.x
            y=int(tokens[3])+self.y
            if(x==0 and y==0):
                return ans
            return ans+str(x)+str(y)

class Library:
    def __init__(self, primitives: list):
        self.primitives_dict = {p.name: p for p in primitives}

    def apply_program(self, p, board_state):
        import copy

        i = board_state
        m = i.copy()
        o = set()

        for txid in p:
            if txid == "out":
                o.update(m)
            elif txid == "clear":
                m = i.copy()
            else:
                t_m = copy.deepcopy(m)
                m = set()
                for elem in t_m:
                    output_elem = self.primitives_dict[txid].apply(elem)
                    if output_elem is not None:
                        m.add(output_elem)
                    else:
                        m.add(elem)
        return o

    def __str__(self):
        return "\n".join([name for name in self.primitives_dict])


def to_shape_creator(shape_name,prefix=""):
    """Creates a Primitive to convert to a shape"""
    new_shape_name=str(prefix)+shape_name
    return Primitive(f"to-{shape_name}", lambda x: (new_shape_name, x[1]))


def shift_creator(board_size=3):
    """Function to generate Primtives with names for shifts in cardinal
    directions and functions that transforms element by changing its
    position according to which direction the shift is for.
    """
    clamp = lambda n, smallest, largest: max(smallest, min(n, largest))
    shift_names = ["right", "left", "up", "down"]
    shift_funcs = [
        lambda x: (x[0], (clamp(x[1][0] + 1, 0, board_size - 1), x[1][1])),
        lambda x: (x[0], (clamp(x[1][0] - 1, 0, board_size - 1), x[1][1])),
        lambda x: (x[0], (x[1][0], clamp(x[1][1] - 1, 0, board_size - 1))),
        lambda x: (x[0], (x[1][0], clamp(x[1][1] + 1, 0, board_size - 1))),
    ]
    return [Primitive(f"shift{name}", f) for name, f in zip(shift_names, shift_funcs)]

import re
def flip(x):
    items = re.findall(r'[0-9]+|[a-z]+', x[0])
    angle=(360-int(items[0]))%360
    return (str(angle)+items[1],x[1])

if __name__ == "__main__":
    import dill
    
    from datagen import BOARD_SIZE

    lib = Library(
        [
            to_shape_creator("square",0),
            to_shape_creator("circle",0),
            to_shape_creator("triangle",0),
            to_shape_creator("delta",0),
            Primitive(
                "flip",
                flip
            ),
            AffineTransform("affine+",5,5),
            AffineTransform("affine-",-5,-5),
            *shift_creator(BOARD_SIZE),
        ]
    )

    # dill.dump(lib, open("../data/libraries/library2.pkl", "wb"))
    # lib = dill.load(open("../data/libraries/library2.pkl", "rb"))

    # print(lib)

    p = ["affine+","out"]
    # board = set([("triangle", "mm"), ("triangle", "tl")])
    board = set([("270s", (0, 0))])
    print(lib.apply_program(p, board))

    # dill.dump(
    #     Library([*shift_creator(BOARD_SIZE)]),
    #     open("data/libraries/shift_library0.pkl", "wb"),
    # )
