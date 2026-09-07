from dataclasses import dataclass

@dataclass
class tempToken():
    score : int
    left_str: str

class Tokenizer():

    def __init__(self, merges_path):

        # read the merges
        with  open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            self.merges = {}

            for i in range(0, len(lines)):
                self.merges[lines[i]] = i


    def tokenize_chunk_merges(self, chunk):

        ar = []

        for i in range(0, len(chunk)):

            if (i != len(chunk)-1):
                pair = chunk[i] +" " + chunk[i+1]

                if pair in self.merges:
                    ar.append(tempToken(score=self.merges[pair], left_str=chunk[i]))
                else:
                    ar.append(tempToken(score=-1, left_str=chunk[i]))
            else:
                ar.append(tempToken(score=-1, left_str=chunk[i]))

        print(ar)



        out = self.naive_sol(ar)
        print(out)
        #print(self.merges)


    def naive_sol(self, ar):


        while len(ar) > 1: # or no more, we early break

            minScore = 99999999
            minIndex = -1

            for i in range(0, len(ar)):
                if ar[i].score != -1 and ar[i].score < minScore:
                    minScore = ar[i].score
                    minIndex = i


            if (minIndex != -1):
                # we have stuff to merge

                # copy over next char
                ar[minIndex].left_str += ar[minIndex+1].left_str

                # delete next token
                new_ar = [0] * (len(ar)-1)
                for g in range(0, len(ar)):
                    if g <= minIndex:
                        new_ar[g] = ar[g]
                    elif g == (minIndex+1):
                        continue
                    else:
                        new_ar[g-1] = ar[g]



                if (minIndex != 0):
                    # update minIndex-1
                    new_ar[minIndex-1].score = self.merges.get(new_ar[minIndex-1].left_str + " " + new_ar[minIndex].left_str, -1)

                if (minIndex == len(new_ar)-1):
                    new_ar[minIndex].score = -1
                else:
                    new_ar[minIndex].score = self.merges.get(new_ar[minIndex].left_str + " " + new_ar[minIndex+1].left_str, -1)

                ar = new_ar
            else:
                return new_ar # early ret cos no more merges
        return new_ar



tokenizer = Tokenizer("/Users/shreybirmiwal/projects/jwlabs/jwLLM/data/merges.txt")
tokenizer.tokenize_chunk_merges("hello")