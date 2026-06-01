import json
import csv
import threading
import queue
from pathlib import Path
from utils.logger import logger

class ResultWriter:
    """
    异步结果持久化工具，将每帧识别结果写入文件。
    """
    def __init__(self, output_path: str, format: str = "json"):
        self.output_path = Path(output_path)
        self.format = format.lower()
        self.queue = queue.Queue()
        self.running = False
        self._thread = None
        
        self.output_path.parent.mkdir(exist_ok=True, parents=True)

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()
        logger.info(f"ResultWriter started. Saving to {self.output_path}")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()
        logger.info("ResultWriter stopped.")

    def write_packet(self, packet):
        """将 Packet 数据放入待写入队列"""
        data = {
            "frame_id": packet.frame_id,
            "ball": packet.ball_coord, # [x, y] or None
            "players": []
        }
        for skel in packet.skeletons:
            data["players"].append({
                "id": skel.get("player_id"),
                "bbox": skel.get("bbox"),
                "strokes": skel.get("stroke_action")
            })
        self.queue.put(data)

    def _write_loop(self):
        if self.format == "json":
            self._write_json()
        elif self.format == "csv":
            self._write_csv()

    def _write_json(self):
        # JSON 存为列表形式
        results = []
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.5)
                results.append(item)
            except queue.Empty:
                continue
        
        with open(self.output_path, 'w') as f:
            json.dump(results, f, indent=4)

    def _write_csv(self):
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "ball_x", "ball_y", "player_id", "bbox"])
            
            while self.running or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.5)
                    fid = data["frame_id"]
                    bx, by = data["ball"] if data["ball"] else (None, None)
                    
                    if not data["players"]:
                        writer.writerow([fid, bx, by, None, None])
                    else:
                        for p in data["players"]:
                            writer.writerow([fid, bx, by, p["id"], p["bbox"]])
                except queue.Empty:
                    continue
