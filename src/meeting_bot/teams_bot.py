"""
Microsoft Teams meeting bot
"""
from typing import Optional

class TeamsBot:
    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        self.email = email
        self.password = password
        self.driver = None
        
    def _init_driver(self):
        """Initialize web driver"""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        
        self.driver = webdriver.Chrome(options=options)
    
    def join_meeting(self, meeting_link: str):
        """Join Teams meeting"""
        self._init_driver()
        
        try:
            self.driver.get(meeting_link)
            time.sleep(5)
            
            from selenium.webdriver.common.by import By
            
            # Click "Join now" button
            join_btn = self.driver.find_element(By.CSS_SELECTOR, "[aria-label='Join now']")
            join_btn.click()
            
            time.sleep(3)
            
            # Turn on microphone
            mic_btn = self.driver.find_element(By.CSS_SELECTOR, "[aria-label='Microphone']")
            mic_btn.click()
            
            return True
            
        except Exception as e:
            print(f"Failed to join Teams meeting: {e}")
            return False
    
    def leave_meeting(self):
        """Leave the meeting"""
        if self.driver:
            self.driver.quit()
            self.driver = None