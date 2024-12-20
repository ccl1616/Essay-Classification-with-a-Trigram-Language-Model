import sys
from collections import defaultdict
import math
import random
import os
import os.path
""" 
Essay Classification with a Trigram Language Mode

Student: Julie Cheng
Instructor: Daniel Bauer
"""

"""
    Using yield can iterate the collection without saving them.
    A form of "lazy avaluation."
"""
def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    # insert n 'START' into a copy of sequence. insert a 'STOP' to the end
    # sentence = []
    # for i in range(n):
    #     sentence.append("START")
    # sentence += sequence
    # sentence.append("STOP")
    sentence = ["START"] * n + sequence + ["STOP"]

    # start from [0], create tuples each with size n.
    result = []
    if n == 1:
      result.append(tuple(sentence[0:1]))
    for i in range(n, len(sentence)):
        # each n-gram is in size n
        temp = sentence[i - n + 1: i + 1]
        result.append(tuple(temp))

    return result


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        # might want to use defaultdict or Counter instead
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)

            # populate dictionaries for n-grams
            for g in unigrams:
                self.unigramcounts[g] += 1
            for g in bigrams:
                self.bigramcounts[g] += 1
            for g in trigrams:
                self.trigramcounts[g] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability

        trigram = (w1, w2, w3)
        p(w3 | w1, w2) = count(trigram) / count(bigram(w1, w2))
        """
        # numerator
        count_trigram = self.trigramcounts.get(trigram, 0)
        # denominator
        bigram = (trigram[0], trigram[1])
        count_bigram = self.bigramcounts.get(bigram, 0)
        # For edge case P(bigram) = 0, return 1/|V|. |V| = size of the lexicon
        if count_bigram == 0:
            v = len(self.lexicon)
            return 1 / v if v > 0 else 0.0

        # Probability of the trigram
        return count_trigram / count_bigram

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability

        bigram = (w1, w2)
        p(w2 | w1) = count(bigram) / count(unigram(w1))
        """
        # numerator
        count_bigram = self.bigramcounts.get(bigram, 0)
        # denominator
        w1 = (bigram[0],)
        count_w1 = self.unigramcounts.get(w1, 0)
        # Probability of the bigram
        return count_bigram / count_w1 if count_w1 > 0 else 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.

        unigram = (w1)
        p(w1) = count(unigram) / Total word in corpus
        Total word in corpus = sum of value of unigram
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        
        # numerator
        count_unigram = self.unigramcounts.get(unigram, 0)
        # denominator
        total_words = sum(self.unigramcounts.values())

        # Probability of the unigram
        return count_unigram / total_words if total_words > 0 else 0.0

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        # start sentence with ("START", "START")
        sentence = ["START", "START"]
        # to create next word, look at raw_trigram_probability for each word.
        # use a random value of probability to select the word. using the accumulate and threshold method.
        # stop when "STOP" is generated or reach t words. number of iteration = t.

        for i in range(t):
            # calculate trigram probability. trigram (w1, w2, w3)
            # sentence = [u, v, w, w1, w2]
            w1, w2 = sentence[-2], sentence[-1]

            # collect all possible w3 and their trigram prob. look up in trigramcounts in order to get possible candidates.
            words, probs = [], []
            for trigram, count  in self.trigramcounts.items():
                if trigram[0] == w1 and trigram[1] == w2 and trigram[2] != "UNK":
                    p = self.raw_trigram_probability(trigram)
                    words.append(trigram[2])
                    probs.append(p)
            
            # randomly pick one
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]     # convert probability in (0, 1)
            # pick a random number in (0, 1). summing event probabilities, pick the word once the sum exceed r.
            r = random.random()     # Return the next random floating-point number in the range 0.0 <= X < 1.0
            cur_prob = 0.0
            w3 = None
            for w, p in zip(words, probs):
                cur_prob += p
                if cur_prob >= r:
                    w3 = w 
                    break
            sentence.append(w3)
            if w3 == "STOP":
                break

        return sentence[2:]     # do not include the "START", "START"

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 

        trigram = (w1, w2, w3)
        bigram = (w2, w3)
        unigram = (w3)
        result 
        = p(w3 | w1, w2) 
        = lambda1 * p(w3 | w1, w2) + lambda2 * p(w3 | w2) + lambda3 * p(w3)
        = lambda1 * raw_trigram_probability(trigram) + lambda2 * raw_bigram_probability(bigram) + lambda3 * raw_unigram_probability(unigram)
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        bigram = (trigram[1], trigram[2])
        unigram = (trigram[2],)
        return lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(bigram) + lambda3 * self.raw_unigram_probability(unigram)
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.

        Use get_ngrams() for trigram.
        Use smoothed_trigram_probability for probabilities.
        Use math.log2(num) for log.

        p(s) = sum( log2(smoothed trigram prob) )
        """
        result = 0.0
        trigrams = get_ngrams(sentence, 3)
        for t in trigrams:
            # calculate smoothed prob for each trigram
            prob = self.smoothed_trigram_probability(t)
            if prob > 0:
                result += math.log2(prob)
            else:
                return float("-inf")
        return result

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.

        perplexity = 2 ^ (-l)
        l = (1/M) * sum(logprob(sentence)) = avg_log_prob
        M = total number of word tokens in corpus
        """
        sum_logprob = 0.0
        M = 0   # total number of tokens

        # process corpus
        for sentence in corpus:
            if len(sentence) == 0:
                continue
            sum_logprob += self.sentence_logprob(sentence)
            M += len(sentence)
        # avoid zero division error
        if M == 0:
            return float("inf")
        
        # compute l
        l = sum_logprob / M
        # compute perplexity by l
        result = 2 ** (-l)

        return result


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    """
    Apply trigram model to text classification. Compute perplexity on each language model on easy essay.
    """
    model1 = TrigramModel(training_file1)   # high
    model2 = TrigramModel(training_file2)   # low

    total = 0
    correct = 0       

    # file 1
    for f in os.listdir(testdir1):
        # read the test essays and compute perplexity
        corpus = corpus_reader(os.path.join(testdir1, f), model1.lexicon)
        pp1 = model1.perplexity(corpus) # should be better
        pp2 = model2.perplexity(corpus)
        # lower pp, better model. pp1 should < pp2
        if pp1 <= pp2:
            correct += 1
        total += 1

    # file 2
    for f in os.listdir(testdir2):
        corpus = corpus_reader(os.path.join(testdir2, f), model2.lexicon)
        pp1 = model1.perplexity(corpus)
        pp2 = model2.perplexity(corpus) # should be better
        # lower pp, better model. pp2 should < pp1
        if pp2 <= pp1:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

if __name__ == "__main__":
    # run the code by "python trigram_model.py hw1_data/brown_train.txt hw1_data/brown_test.txt"
    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing logprob
    model = TrigramModel(sys.argv[1])
    test_corpus = list(corpus_reader(sys.argv[2], model.lexicon))

    sample_sentence = test_corpus[0]  # Using the first sentence from the test set
    log_prob = model.sentence_logprob(sample_sentence)
    print(f"Log Probability of the first sentence: {log_prob}")
    
    # Training perplexity. Should be smaller than Testing.
    # dev_corpus = corpus_reader(sys.argv[1], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(f"Training Perplexity: {pp}")

    # Testing perplexity
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(f"Testing Perplexity: {pp}")

    # Test unigram, bigram, trigram probabilities
    sample_sentence = test_corpus[0]  # First sentence from the test set
    print("Sample sentence:", sample_sentence)

    # Test unigram probability
    # unigram = (sample_sentence[0],)  # First word as a unigram
    unigram = ('the',)
    unigram_prob = model.raw_unigram_probability(unigram)
    print(f"Raw Unigram Probability of '{unigram}': {unigram_prob}")
    # printing
    # numerator
    count_unigram = model.unigramcounts.get(unigram, 0)
    # denominator
    total_words = sum(model.unigramcounts.values())
    print("unigram count", count_unigram, total_words)

    # Test bigram probability
    if len(sample_sentence) > 1:
        # bigram = (sample_sentence[0], sample_sentence[1])
        bigram = ('START','the')
        bigram_prob = model.raw_bigram_probability(bigram)
        print(f"Raw Bigram Probability of '{bigram}': {bigram_prob}")
        print("bigram count", model.bigramcounts.get(bigram, 0))

    # Test trigram probability
    if len(sample_sentence) > 2:
        # trigram = (sample_sentence[0], sample_sentence[1], sample_sentence[2])
        trigram = ('START','START','the')
        trigram_prob = model.raw_trigram_probability(trigram)
        print(f"Raw Trigram Probability of '{trigram}': {trigram_prob}")
        print("trigram count", model.trigramcounts.get(trigram, 0))

    # Essay scoring experiment: 
    acc = essay_scoring_experiment(
        "hw1_data/ets_toefl_data/train_high.txt", 
        "hw1_data/ets_toefl_data/train_low.txt", 
        "hw1_data/ets_toefl_data/test_high", 
        "hw1_data/ets_toefl_data/test_low"
    )
    print("essay_scoring:", acc)

    print( model.generate_sentence() )
    print( model.generate_sentence() )
    print( model.generate_sentence() )
    print( model.generate_sentence() )
    print( model.generate_sentence() )