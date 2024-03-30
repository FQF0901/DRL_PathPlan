done = 0

a = 0
b = 1
c = 1

done |= (a << 0) | (b << 1)| (c << 2)

print(bin(done))  # 输出 done 的二进制表示