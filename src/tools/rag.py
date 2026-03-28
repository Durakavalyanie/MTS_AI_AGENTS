from __future__ import annotations
import json
from pathlib import Path

def load_best_trajectories(best_trajectories_dir: Path, max_runs: int = 3) -> str:
    if not best_trajectories_dir.exists():
        return ""
    
    summaries = []
    json_files = list(best_trajectories_dir.glob("**/*.json"))
    
    for json_file in json_files[:max_runs]:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            run_name = json_file.stem
            best_score = None
            key_steps = []
            
            for event in data:
                ev_type = event.get("event_type")
                payload = event.get("payload", {})
                
                if ev_type == "kaggle_submission":
                    score = payload.get("public_score")
                    if score is not None:
                        if best_score is None or score < best_score:
                            best_score = score
                
                if ev_type == "agent_completed_task":
                    agent = payload.get("agent")
                    content = payload.get("content", "")
                    if "[TOOL_CALL]send_message" in content:
                        try:
                            msg_start = content.find("{")
                            msg_end = content.rfind("}") + 1
                            if msg_start != -1 and msg_end != -1:
                                msg_json = json.loads(content[msg_start:msg_end])
                                key_steps.append(f"- {agent}: {msg_json.get('message', '')}")
                        except:
                            pass
            
            summary = f"### Past Run: {run_name}\n"
            if best_score is not None:
                summary += f"Best Score: {best_score}\n"
            summary += "Key Steps:\n" + "\n".join(key_steps[-5:]) 
            summaries.append(summary)
            
        except Exception as e:
            continue
            
    if not summaries:
        return ""
        
    return "\n## PAST SUCCESSFUL APPROACHES (RAG)\n" + "\n\n".join(summaries)
