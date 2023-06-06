cuda-based symbol address vanity gen
============================

This is cuda-based Symbol address vanity generator.

Chacha20 generator is used as source of randomness and it is seeded with OS random source ([/dev/urandom](https://en.wikipedia.org/wiki//dev/random) or [BCryptOpenAlgorithmProvider](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptopenalgorithmprovider)).

~~You wouldn't steal~~ Open source all the things
-------------------------------------------------

 * chacha20 is borrowed from [DPBayes/jax-chacha-prng](https://github.com/DPBayes/jax-chacha-prng)
 * ED25519 scalar multiplication is **not optimized**, it's taken from [ChorusOne/solanity](https://github.com/ChorusOne/solanity).
 * SHA2-512 also comes from [ChorusOne/solanity](https://github.com/ChorusOne/solanity) which is taken from [libtom/libtomcrypt](https://github.com/libtom/libtomcrypt)
 * sha3-256 comes from [skapix/sha3.git](https://github.com/skapix/sha3.git)

On my rtx generator achieves 4M+ keys per second.

```
device: NVIDIA GeForce RTX 3070
 : maxThreadsPerBlock: 1024
 : multiProcessorCount: 46
 *        sha will run with 46 640 (46)
 * scalarmult will run with 46 256 (46)
 *       sha3 will run with 736 25
 *       ripe will run with 92 320 (92)
 *   matching will run with 736 32
...
took 69.897095 294400000 (4.211906 mhps, 4211906.088515)
```

Right now there's only pretty ugly `compile.bat` for windows, PRs for *nix makefile are welcome.

Sample output
-------------

```
c:\yet-another-vanity-gen> cuda-symbol-vanity-gen.exe ALANIS 50000

pattern ex network byte, as hex:
 > 00000000: 16 06 A2 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
device: NVIDIA GeForce RTX 3070
 : maxThreadsPerBlock: 1024
 : multiProcessorCount: 46
RUNNING SELF-CHECKS
 *  chacha20 block self-check SUCCESS
 *  sha2-512 self-check SUCCESS
 *  scalarmult (priv-to-pub) self-check SUCCESS
 *  sha3-256 self-check SUCCESS
 *  ripemd-160 self-check SUCCESS
PREPARING CONFIGURATION
 *        sha will run with 46 640 (46)
 * scalarmult will run with 46 256 (46)
 *       sha3 will run with 736 25
 *       ripe will run with 92 320 (92)
 *   matching will run with 736 32
[38937 .   298] match counters (4460166,  17384,     64,      0,      0)
priv: E0 73 F1 C1 FE 32 9F AD 77 41 67 FD 96 CA 5F A6 2C 43 77 65 84 77 AF 4B 07 FA A5 64 4B 9D 4B 19
 pub: 08 7D E4 49 DC 14 1B E4 77 5F F0 7F DF B0 EA 71 BA 69 0C 6C E0 25 A2 22 7E 85 DB 24 1E 4D 6B EF
ripe: 16 06 A2 40 77 51 6F 81 5B A0 BA 67 D6 13 99 7E 46 CE 0E F2
[49998 .   268] match counters (5727516,  22253,     82,      1,      0)
took 355.775361 1472000000 (4.137442 mhps, 4137442.225071)
```
