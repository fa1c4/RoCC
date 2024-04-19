import random
import os


for k in range(10):
    message = ''
    for i in range(1000000):
        message += str(random.randint(0, 1))    

    print(message)
    message_file_path = os.path.join('./temp/messages', 'message-{}.txt'.format(k))
    with open(message_file_path, 'w', encoding='utf-8') as f:
        f.write(message)