# Commands
## Windows

```shell
C:\Users\oscar\AppData\Local\Programs\Python\Python311\python.exe setup.py build_ext --inplace
```

## Linux

```shell
python3 setup.py build_ext --inplace
```

```shell
python3 testing_interface.py p 123456789 0 --algo lcg | dieharder -g 200 -d <ID> > out.txt
```

```shell
python3 testing_interface.py f 123456789 0 --algo lcg > out.bin
```

```shell
dieharder -g 201 -f in.bin -d <ID> > out.txt 2>&1
```

```bash
xxd -b file.bin | head -n 20
```

---
> For testing:

```shell  
python3 testing_interface.py f 123456789 0 --total 20 --algo lcg --debug 
```  


---
> Others:
```shell
gcc test_from_pipe.c -o test_from_pipe \
  -I/usr/local/include \
  -L/usr/local/lib \
  -ltestu01 -lprobdist -lmylib -lm
```

```shell
python3 testing_interface.py p 123456789 0 --algo lcg | ./test_from_pipe > results.txt
```

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

# Dockerizing

```shell
docker build -t prng-bench .
```

```shell
docker run -it --rm prng-bench
```
