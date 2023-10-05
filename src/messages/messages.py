from uagents import Model

class Temperature(Model):
    value: float
    def printsomething():
        print("I am the debugger function")

class Alert(Model):
    message: str
    def printsomething():
        print("I am the debugger function")

