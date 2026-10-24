import math
import random
import json

random.seed(0) # same number whenever i run so that my ouput can't be change

D = 8 #how biger my EMbedding is 
LR = 0.3
EPOCHS = 400      # FIXED 1: was EPOCH, but the loop below uses EPOCHS

#List of the  sentence 

corpus = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat ran to the dog",
    "the dog ran to the cat",
    "a car drove on the road",
    "the car stopped at the road",
]

# Split the sentenc as per the space 

word = sorted(set(w for line in corpus for w in line.split() ))

#give the index for the each word 
vocab = {w: i for i , w in enumerate(word)}

##again we flip them into the id word
in_vocab = {i : w for w , i in vocab.items()}

sentence = [[vocab[w] for w in line.split()] for line in corpus]
V = len(vocab)
print(vocab)
print(len(vocab))

#step 2 
#pair the every current id 
pairs = []        # FIXED 2: was pair; the rest of the code uses pairs
for s in sentence:
    for i in range(len(s)-1  ):
        pairs.append((s[i] , s[i + 1]))
print(pairs)

# creat random matrix 

def random_matrix( n_row, n_col, std =0.1):
    matrix = []
    for i in range(n_row):

        row = []
        for c in range(n_col):
            number = random . gauss(0, std)

            row.append(number)

        matrix.append(row)
    return matrix
E = random_matrix(V, D)
W = random_matrix(D, V)
b = [0.0] * V

#soft max 

def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]

    total = sum(exps)
    return [e / total for e in exps]

#print(softmax([2.0, 1.0, 0.1]))

#Softmax turns raw scores into probabilities: all positive, adding up to 1, with the biggest score
#  getting the biggest share. The LLM needs them 
# to pick the next word, measure its mistake, and control how creative the text is.

def forward(cur_id):
    x = E[cur_id]
    logits = []
    for j in range(V):                       # FIXED 3: was "for j in V" (a number can't be looped)
        total = 0
        for k in range(D):                   # FIXED 4: was "for k in D"
            total += x[k] * W[k][j]          # FIXED 5: was x[k][j]; the weights are in W
        total += b[j]
        logits . append(total)
    probs = softmax(logits)
    
    return x , probs
#step 5 
def loss_far (cur_id, next_id):
    x , probs = forward(cur_id)
    p = probs[next_id]                       # FIXED 6: was probs(next_id); a list uses [ ] not ( )
    return -math.log(p)

#avg loss function
def average_loss ():                         # FIXED 7: was avg_loss; the code below calls average_loss
    total = 0
    for cur_id, next_id in pairs:            # FIXED 8: was "in range(pair)"; loop over the pairs list
        total += loss_far(cur_id, next_id)
    return total / len(pairs)                # FIXED 9: was len(pair)

# step 6 back forward 
# FIXED 10: this code was loose at the top of the file, where next_id, probs, x and cur_id
# do not exist yet. It must live inside functions that receive them.
def gradients(cur_id, next_id):
    x, probs = forward(cur_id)               # we need the card x and the probabilities

    dlogit = []
    for j in range(V):                       # FIXED 11: was range(v); the name is V (capital)
        if j == next_id :
            dlogit . append(probs[j]- 1.0)   # FIXED 12: was "1 .0" (a syntax error)
        else :
            dlogit.append(probs[j])

    #slop 
    dW = []
    for k in range(D):
        row = []
        for j in range(V):
            row.append(x[k] * dlogit[j])     # card number k  ×  slope of word j's score
        dW.append(row)

    #the slop for the each bias 
    db = []
    for j in range(V):
        db.append(dlogit[j])

    #the slop for each numer card 
    dx = []
    for k in range(D):
        total = 0
        for j in range(V):
            total += W[k][j] * dlogit[j]     # card number k feeds into ALL the scores
        dx.append(total)
    return dW, db, dx

# the updated one 
def train_step(cur_id, next_id):
    dW, db, dx = gradients(cur_id, next_id)
    for k in range(D):
        for j in range(V):
            W[k][j] -= LR * dW[k][j]          # new = old - LR x slope
    for j in range(V):
        b[j] -= LR * db[j]
    for k in range(D):
        E[cur_id][k] -= LR * dx[k]            # only the row that was used

# FIXED 13: these three functions were missing, but the lines below call them
def cosine(a, c):
    dot = 0
    len_a = 0
    len_c = 0
    for i in range(len(a)):
        dot += a[i] * c[i]
        len_a += a[i] * a[i]
        len_c += c[i] * c[i]
    return dot / (math.sqrt(len_a) * math.sqrt(len_c))

def similarity(word1, word2):
    return cosine(E[vocab[word1]], E[vocab[word2]])

def nearest(w, n=3):
    scored = []
    for other in vocab:
        if other != w:
            scored.append((similarity(w, other), other))
    scored.sort(reverse=True)
    result = []
    for sim, other in scored[:n]:
        result.append(other)
    return result


print("vocab_size V = %d, pairs = %d, D = %d" % (V, len(pairs), D))
print("starting loss = %.3f  (guessing = ln(V) = %.3f)" % (average_loss(), math.log(V)))
print("BEFORE: sim(cat,dog) = %.3f, sim(cat,car) = %.3f\n" % (similarity("cat", "dog"), similarity("cat", "car")))
 
for epoch in range(1, EPOCHS + 1):
    random.shuffle(pairs)                       # mix the order each epoch
    for cur_id, next_id in pairs:
        train_step(cur_id, next_id)             # one nudge per pair
    if epoch in (1, 10, 50, 100, 200, 400):
        print("epoch %3d   loss = %.3f" % (epoch, average_loss()))
 
final_loss = average_loss()
sim_cat_dog = similarity("cat", "dog")
sim_cat_car = similarity("cat", "car")
print("\nAFTER : sim(cat,dog) = %.3f, sim(cat,car) = %.3f" % (sim_cat_dog, sim_cat_car))
print("closest to 'cat':", nearest("cat"))
 
print("\nSUCCESS CHECKS")
print("  loss below ln(V)     :", "PASS" if final_loss < math.log(V) else "FAIL")
print("  cat closer to dog    :", "PASS" if sim_cat_dog > sim_cat_car else "FAIL")
 
