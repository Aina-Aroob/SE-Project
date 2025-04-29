from fastapi import FastAPI, WebSocket
from bat_detection import process_input, predict_trajectory
import uvicorn
import json

app = FastAPI|()

@app.websocket('/detect')
async def detect_collision(websocket:WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            result = process_input(data)
            print(json.dumps(result, indent=2))
            
            # If collision occurred, visualize the new trajectory
            if result["collision"]["collision"] and result["trajectory"]["updated"]:
                new_trajectory = predict_trajectory(
                    data["detection"]["center"],
                    result["trajectory"]["velocity"]
                )
                steps_dict = {f"Step {i}": pos for i, pos in enumerate(new_trajectory)}
                json_output = json.dumps(steps_dict, indent=2)
                combined = {**result, "new_trajectory_steps": steps_dict}
                return json.dumps(combined)
            else:
                return result
    except Exception as e:
        print(e)



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)