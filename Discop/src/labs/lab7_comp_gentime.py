import json 


with open('./jsons/length_stego_times.json', 'r', encoding='utf-8') as f:
    json_data = json.loads(f.read())

max_transrate = 0.0
max_rate_length = 100
min_transrate = 1000.0
min_rate_length = 100
for length, values in json_data.items():
    encode_time = values[0]
    encode_bits = values[1]
    transrate = encode_bits / encode_time
    if transrate > max_transrate:
        max_rate_length = length
        max_transrate = transrate
    
    if transrate < min_transrate:
        min_transrate = transrate
        min_rate_length = length

    print('token per second: ', int(length) / encode_time)

print('max_transrate: ', max_transrate)
print('max_rate_length: ', max_rate_length)

print('min transrate: ', min_transrate)
print('min_rate_length: ', min_rate_length)