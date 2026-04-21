"""
Zoom meeting bot for automatic joining and recording
"""
from typing import Optional
import time

class ZoomBot:
    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        self.email = email
        self.password = password
        self.driver = None
        
    def _init_driver(self):
        """Initialize web driver"""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument('--headless')  # Run in background
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(options=options)
    
    def join_meeting(self, meeting_url: str, password: Optional[str] = None):
        """Join Zoom meeting"""
        self._init_driver()
        
        try:
            # Navigate to meeting URL
            self.driver.get(meeting_url)
            time.sleep(3)
            
            # Enter password if provided
            if password:
                from selenium.webdriver.common.by import By
                pwd_input = self.driver.find_element(By.ID, "input-for-password")
                pwd_input.send_keys(password)
                
                join_btn = self.driver.find_element(By.ID, "join-meeting")
                join_btn.click()
            
            time.sleep(2)
            
            # Join audio
            from selenium.webdriver.common.by import By
            try:
                audio_btn = self.driver.find_element(By.CLASS_NAME, "join-audio")
                audio_btn.click()
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"Failed to join meeting: {e}")
            return False
    
    def leave_meeting(self):
        """Leave the meeting"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def record_meeting(self, duration_minutes: int, output_path: str):
        """Record meeting audio"""
        from ..audio.recorder import SystemAudioRecorder
        
        recorder = SystemAudioRecorder()
        recorder.record_system_audio(duration_minutes * 60, output_path)