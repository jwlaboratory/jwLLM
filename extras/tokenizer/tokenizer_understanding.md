This is to understand how the byte unicode works in the tokenizer

```c
#include <stdio.h>

int main() {
    // The fire emoji UTF-8 byte sequence
    printf("%s\n", "\xF0\x9F\x94\xA5");
    return 0;
}
```


The fire emoji is: (in utf-8, unicode)
F0 9F 94 A5

in big endian, the most significant is at lowest address. in little endian, the least significant (a5) is at the lowest mem address.
ENDINENSSS DOES NOT APPLY TO UTF8!!!! its just sequnece of bytes
they are stores in an array


utf 8 is variable lenght

F0 9F 94 A5 == 1111-0000 1001-1111 1001-0100 1010-0101
U+10FFFF	11110uvv	10vvwwww	10xxxxyy	10yyzzzz

payload

this means payload = 
000, 01-1111, 01-0100, 10-0101




# step 2: do the byte2unicode
1111-0000.   1001-1111.   1001-0100.   1010-0101 
f0, 9f, 94, a5

f0 = dec 240 = nice
we want to keep 240 as 240
so we rencode as one UTF

[240   159   148   165]

2) do our mapping
if trouble, add the n

[240, 321, 310, 165]

3) each number, which used to be meant to be interpreted in variable length is now interepreted liek 1 or 2 bytes
so 240 becomes bytes C3 B0 
in unicode, 240 needs 2 byte template 110xxxxx   10xxxxxx
so we fill in and get C3 B0

so then now we have [[c3 b0], [c5 81], [c4 b6], [c2 a5]]
then we do merges

final answer outputs
[c3b0c581], [c4b6] [c2a5]
with first 2 glued to



steps
1. input
2. for each byte, remap to multiple bytes long and remove bad ones
3. do merges now with these new characters