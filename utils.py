@staticmethod
def print_node(g):
    print(g.label)
    print(g.conditions)
    print(g.cpt)
    print('---------------------------------------')

@staticmethod
def build_dict(conditions):
    dict = {}
    for c in conditions:
        value = 1 if len(c) == 2 else 0
        key = c[-1]
        dict[key] = value

    return dict





def main():
    print('nothing')

if __name__ == '__main__':
    main()