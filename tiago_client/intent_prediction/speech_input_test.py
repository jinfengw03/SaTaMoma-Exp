#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Speech Input Test Script
语音输入测试脚本
"""

import rospy
from std_msgs.msg import String
import sys

class SpeechInputTest:
    def __init__(self):
        rospy.init_node('speech_input_test_node')
        
        # 发布器
        self.speech_pub = rospy.Publisher('/speech_input', String, queue_size=10)
        
        # 预设测试语句
        self.test_phrases = {
            '1': "I am hungry",
            '2': "I am thirsty",
            '3': "I want to drink water",
            '4': "Please hand me the book",
            '5': "Can you get me the red cup",
            '6': "I need the remote control",
            '7': "Bring me that tool",
            '8': "Pick up the apple",
            '9': "I want to eat something",
            '0': "Put it on the table"
        }
        
        rospy.loginfo('Speech Input Test Node initialized')
        rospy.loginfo('Publishing to topic: /speech_input')
        
    def publish_speech(self, text):
        """发布语音文本"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        rospy.loginfo(f'Published: "{text}"')
    
    def show_menu(self):
        """显示菜单"""
        print("\n" + "="*60)
        print("Speech Input Test - 语音输入测试")
        print("="*60)
        print("\nQuick Test Phrases (快速测试语句):")
        for key, phrase in sorted(self.test_phrases.items()):
            print(f"  [{key}] {phrase}")
        print("\n  [c] Custom input (自定义输入)")
        print("  [a] Auto test all (自动测试所有)")
        print("  [q] Quit (退出)")
        print("="*60)
    
    def auto_test(self):
        """自动测试所有预设语句"""
        print("\n[Auto Test] Testing all phrases...")
        rate = rospy.Rate(0.2)  # 每5秒一个
        
        for key in sorted(self.test_phrases.keys()):
            if rospy.is_shutdown():
                break
            phrase = self.test_phrases[key]
            print(f"\n[{key}] Testing: {phrase}")
            self.publish_speech(phrase)
            rate.sleep()
        
        print("\n[Auto Test] Complete!")
    
    def run_interactive(self):
        """交互式运行"""
        while not rospy.is_shutdown():
            self.show_menu()
            
            try:
                choice = input("\nEnter your choice: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
            
            if choice == 'q':
                print("Goodbye!")
                break
            
            elif choice == 'a':
                self.auto_test()
            
            elif choice == 'c':
                try:
                    custom_text = input("Enter custom speech text: ").strip()
                    if custom_text:
                        self.publish_speech(custom_text)
                    else:
                        print("Empty input, skipped.")
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
            
            elif choice in self.test_phrases:
                phrase = self.test_phrases[choice]
                self.publish_speech(phrase)
            
            else:
                print(f"Invalid choice: {choice}")
            
            rospy.sleep(0.5)
    
    def run_command_line(self, text):
        """命令行模式：直接发布文本"""
        self.publish_speech(text)
        rospy.sleep(1)

def main():
    try:
        test_node = SpeechInputTest()
        
        # 检查命令行参数
        if len(sys.argv) > 1:
            # 命令行模式：直接发布参数
            speech_text = ' '.join(sys.argv[1:])
            print(f'Command-line mode: Publishing "{speech_text}"')
            test_node.run_command_line(speech_text)
        else:
            # 交互式模式
            test_node.run_interactive()
    
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nInterrupted by user")

if __name__ == '__main__':
    main()
