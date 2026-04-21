"""
Meeting scheduler for automatic joining
"""
from typing import List, Dict
from datetime import datetime
import threading
import time

class MeetingScheduler:
    def __init__(self):
        self.scheduled_meetings = []
        self.running = False
        self.thread = None
        
    def add_meeting(self, meeting_url: str, start_time: datetime, 
                    duration_minutes: int, bot_instance, meeting_id: str = None):
        """Schedule a meeting"""
        meeting = {
            'id': meeting_id or str(len(self.scheduled_meetings)),
            'url': meeting_url,
            'start_time': start_time,
            'duration': duration_minutes,
            'bot': bot_instance,
            'joined': False
        }
        self.scheduled_meetings.append(meeting)
        return meeting['id']
    
    def start(self):
        """Start the scheduler"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _run(self):
        """Main scheduler loop"""
        while self.running:
            now = datetime.now()
            
            for meeting in self.scheduled_meetings:
                if not meeting['joined']:
                    # Join 5 minutes before start
                    time_diff = (meeting['start_time'] - now).total_seconds()
                    
                    if -300 <= time_diff <= 60:  # 5 min before to 1 min after
                        print(f"Joining meeting {meeting['id']}")
                        meeting['bot'].join_meeting(meeting['url'])
                        meeting['joined'] = True
                        
                        # Schedule leaving
                        leave_time = meeting['start_time'].timestamp() + meeting['duration'] * 60
                        threading.Timer(
                            max(0, leave_time - time.time()),
                            self._leave_meeting,
                            args=[meeting]
                        ).start()
            
            time.sleep(30)  # Check every 30 seconds
    
    def _leave_meeting(self, meeting):
        """Leave a meeting"""
        print(f"Leaving meeting {meeting['id']}")
        meeting['bot'].leave_meeting()
    
    def get_scheduled(self) -> List[Dict]:
        """Get all scheduled meetings"""
        return self.scheduled_meetings
    
    def cancel_meeting(self, meeting_id: str):
        """Cancel a scheduled meeting"""
        self.scheduled_meetings = [m for m in self.scheduled_meetings if m['id'] != meeting_id]