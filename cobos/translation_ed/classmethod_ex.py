class foo:
    def __init__(self, a):
        self.a = a

    @classmethod
    def manyfoos(cls, a_list):
        return [cls(a) for a in a_list]
    
    def __repr__(self) -> str:
        return f'<foo (a = {self.a})>'