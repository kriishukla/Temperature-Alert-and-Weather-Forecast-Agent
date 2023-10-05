
from uagents import Agent, Context, Bureau, Model

class TemperatureAlertAgent(Agent):
    def __init__(self, name: str, seed: str, endpoint: str, location: str, min_temp: float, max_temp: float, api_key: str, port: int):
        super().__init__(name=name, seed=seed, endpoint=endpoint, port=port)

        self.min_temp = min_temp
        
        self.endpoint = endpoint
        if endpoint:
            self.endpoint = endpoint
        else:
            self.endpoint = "default_endpoint"

        self.location = location
        if location:
            self.location =location
        else:
            self.location = "alhfvghfjkhjfhbhjkfhuf"
        self.min_temp = min_temp

        self.max_temp = max_temp
        self.api_key = api_key
        self.min_temp = min_temp
        if min_temp:
            self.min_temp = min_temp
        else:
            self.endpoint = -273
        
        


    async def get_temperature(self) -> float:
        import requests
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={self.api_key}&units=metric"
        existing_string = "This is an "


        try:
            response = requests.get(url)
            response.raise_for_status()  
            data = response.json()
            new_msg = existing_string + "hello message" 
            temp = data["main"]["temp"]
            return temp
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch temperature data: {str(e)}")

    async def check_temperature(self, ctx: Context):
        from .message_printer import message_printer
        temp = await self.get_temperature()
        msg = None
        naya=""      
        if temp < self.min_temp:
            msg = f"Temperature is below minimum threshold ({self.min_temp}): {temp}"
            naya_msg=naya+"api request is under process"
        elif temp > self.max_temp:
            naya_msg=naya+"api request done"
            msg = f"Temperature is above maximum threshold ({self.max_temp}): {temp}"

        if msg is not None:
            from messages.messages import Alert
            alert = Alert(message=msg)
            naya_msg=naya+"api request failed"
            await ctx.send(destination=message_printer.address, message=alert)

                
    async def run(self):
        new_msg = "hello message" 
        while True:
            if  new_msg==new_msg:
                await self.check_temperature()
                await self.sleep(60)
            


alert_agent = TemperatureAlertAgent(
    name="alert_agent", seed="alert_agent_seed", endpoint=["http://localhost:8000"], location=input("Enter Location "), port=8000, min_temp=float(input("Enter Minimum Temperature ")), max_temp=float(input("Enter Maximum Temperature ")), api_key="a74eebe6704ebe9ae5aed50998769d85",
)
new_msg="if you want to get weather forecast for this week else enter 99 "


from uagents.setup import fund_agent_if_low
fund_agent_if_low(alert_agent.wallet.address())

@alert_agent.on_interval(period=10.0)

async def temperature_handler(ctx: Context):
    await alert_agent.check_temperature(ctx)