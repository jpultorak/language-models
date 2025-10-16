from lab1.sentence_probability import sentence_prob
from itertools import permutations


TOP_N = 15

def words_to_sentence(words):
    words[0] = words[0].capitalize()
    s = ' '.join(words)
    s += "."
    return s

def all_sentences(words):
    words = [w.lower() for w in words]
    perms = set(permutations(words))
    sentences = [words_to_sentence(list(w)) for w in perms]
    return sentences


def main(words):
    sentences = all_sentences(words)
    rank = [(sentence_prob(s), s) for s in sentences]

    rank.sort(reverse=True)

    print(f"TOP {TOP_N} most likely sentences")
    for i, (prob, s) in enumerate(rank):
        print(f"{i+1}. ({prob}): {s}")
        if i + 1 >= TOP_N:
            break

    print("==================")

    print(f"TOP {TOP_N} least likely sentences")
    for i, (prob, s) in enumerate(reversed(rank)):
        print(f"{i+1}. ({prob}): {s}")
        if i + 1 >= TOP_N:
            break

if __name__ == "__main__":
    words = ['wiewiórki', 'w', 'parku', 'zaczepiają', 'przechodniów']
    words = ['babuleńka', 'miała', 'dwa', 'rogate','koziołki']
    words = ['małpki', 'kradną', 'złoto', 'kotki', 'banany', 'i', 'i']
    # words = ['hej', 'jestem', 'żólwiem']
    main(words)

