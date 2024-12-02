gb = 1024
dp_size = 8
prod = gb // dp_size 
mb = [d for d in range(1, prod) if prod % d == 0]
gas = [int(gb/(x*dp_size)) for x in mb ]
#gas = gb // (mb * dp_size)

print(prod)
print(mb)
print(gas)
