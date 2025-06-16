from dataclasses import dataclass
import base64
import json
import numpy as np
import requests
from rich import print


@dataclass
class ServerArgument:
    port: int = 9500
    url: str = "10.xx.xx.xx" #input url here


class Server:
    def __init__(self, url: str, port: int):
        self.url = url
        self.port = port
        self.base = f"{self.url}:{self.port}"
        self.check()

    def check(self):
        print("[green]Checking server status[/green]")
        try:
            response = requests.get(f"{self.url}:{self.port}/status")
            if response.status_code == 200:
                print("[green]Server is running[/green]")
            else:
                print("[red]😭Server is not running. Please start the Server![/red]")
        except requests.exceptions.RequestException as e:
            print(f"[red]😭Error connecting to server: {e}[/red]")

    def get_initial_text(self) -> dict:
        try:
            response = requests.get(f"{self.base}/initial_text")
            if response.status_code == 200:
                data = response.json()
                print(f"[green]Initial text: {data.get('text')}[/green]")
                return data
            else:
                print(f"[red]Failed to get initial text: {response.status_code} {response.text}[/red]")
        except requests.exceptions.RequestException as e:
            print(f"[red]Error connecting to server: {e}[/red]")

    def reset(self):
        print("[green]Resetting server[/green]")
        # return
        # Reset the server here
        try:
            # payload = {"device": device}
            response = requests.post(f"{self.url}:{self.port}/reset", json={"device": "cuda:0"})
            if response.status_code == 200:
                data = response.json()
                print("[green]Server reset successfully[/green]")
                return data
            else:
                print("[red]😭Failed to reset server[/red]")
        except requests.exceptions.RequestException as e:
            print(f"[red]😭Error connecting to server: {e}[/red]")

    def pause(self):
        print("[green]Pause Agent[/green]")
        # return
        # Reset the server here
        try:
            # payload = {"device": device}
            response = requests.post(f"{self.url}:{self.port}/pause")
            if response.status_code == 200:
                data = response.json()
                print("[green]Pause agent successfully[/green]")
                return data
            else:
                print("[red]😭Failed to pause agent[/red]")
        except requests.exceptions.RequestException as e:
            print(f"[red]😭Error connecting to server: {e}[/red]")

    def resume(self):
        print("[green]resume Agent[/green]")
        try:
            # payload = {"device": device}
            response = requests.post(f"{self.url}:{self.port}/resume")
            if response.status_code == 200:
                data = response.json()
                print("[green]resume agent successfully[/green]")
                return data
            else:
                print("[red]😭Failed to resume agent[/red]")
        except requests.exceptions.RequestException as e:
            print(f"[red]😭Error connecting to server: {e}[/red]")

    def send_text(self, text: str, task: str) -> dict:
        print(f"[green]Sending text command to server: {text}[/green]")
        try:
            payload = {"text": text, "task": task}
            response = requests.post(f"{self.base}/send_text", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"[green]Received response: {data.get('response')}[/green]")
                return data
            else:
                print(f"[red]😭 Failed to send text: {response.status_code} {response.text}[/red]")
        except requests.exceptions.RequestException as e:
            print(f"[red]😭 Error connecting to server: {e}[/red]")

    def receive_obs(self) -> str:
       
        # print("[green]Requesting observation[/green]")
        try:
            
            r = requests.get(f"{self.base}/get_obs")
            if r.status_code == 200:
                resp_data = r.json()
                return resp_data.get("observation")
            else:
                print(f"[red]Obs failed: {r.status_code} {r.text}[/red]")
        except Exception as e:
            print(f"[red]Error: {e}[/red]")

    def receive_text(self) -> dict:
       
        print("[green]Requesting status text from server[/green]")
        try:
            response = requests.get(f"{self.base}/receive_text")
            if response.status_code == 200:
                data = response.json()
                print(f"[green]Status: {data.get('text')}[/green]")
                return data
            else:
                print(f"[red]😭 Failed to receive text: {response.status_code} {response.text}[/red]")
        except requests.exceptions.RequestException as e:
            print(f"[red]😭 Error connecting to server: {e}[/red]")

    def check_gpu(self) -> dict:
       
        print("[green]Checking GPU status on server[/green]")
        try:
            response = requests.get(f"{self.base}/gpu")
            if response.status_code == 200:
                data = response.json()
                print(f"[green]GPU info: {data}[/green]")
                return data
            else:
                print(f"[red]😭 Failed to check GPU: {response.status_code} {response.text}[/red]")
        except requests.exceptions.RequestException as e:
            print(f"[red]😭 Error connecting to server: {e}[/red]")

    def decode_image(b64_string, dtype=np.uint8) -> np.ndarray:
        decoded = base64.b64decode(b64_string)
        return np.frombuffer(decoded, dtype=dtype)
