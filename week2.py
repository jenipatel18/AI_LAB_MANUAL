import heapq
import string
import urllib.parse
import os
import sys

def preprocess(sentence):
    s = sentence.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s.strip()

def clean_filepath(filepath):
    if filepath.startswith('file:///'):
        filepath = filepath[8:]
    elif filepath.startswith('file://'):
        filepath = filepath[7:]
    filepath = urllib.parse.unquote(filepath)
    filepath = os.path.normpath(filepath)
    return filepath

def read_document_from_file(filepath):
    sentences = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(preprocess(line))
    except FileNotFoundError:
        print(f"Error: File not found -> {filepath}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)
    return sentences

class PlagiarismEnv:
    def __init__(self, doc1, doc2):
        self.doc1 = doc1
        self.doc2 = doc2
        self.n1 = len(doc1)
        self.n2 = len(doc2)

    def edit_distance(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    def is_goal(self, state):
        i, j, _ = state
        return i == self.n1 and j == self.n2

    def get_next(self, state):
        i, j, cost = state
        moves = []
        if i < self.n1 and j < self.n2:
            dist = self.edit_distance(self.doc1[i], self.doc2[j])
            moves.append((i + 1, j + 1, cost + dist, ("align", self.doc1[i], self.doc2[j])))
        if i < self.n1:
            moves.append((i + 1, j, cost + len(self.doc1[i]), ("skip_doc1", self.doc1[i])))
        if j < self.n2:
            moves.append((i, j + 1, cost + len(self.doc2[j]), ("skip_doc2", self.doc2[j])))
        return moves

    def heuristic(self, i, j):
        return abs((self.n1 - i) - (self.n2 - j))

class AStarAgent:
    def __init__(self, env):
        self.env = env
    def a_star(self):
        start = (0, 0, 0)
        frontier = [(0, start, [])]
        visited = {}
        while frontier:
            _, (i, j, cost), path = heapq.heappop(frontier)
            if self.env.is_goal((i, j, cost)):
                return path, cost
            if (i, j) in visited and visited[(i, j)] <= cost:
                continue
            visited[(i, j)] = cost
            for nxt in self.env.get_next((i, j, cost)):
                ni, nj, ncost, action = nxt
                f = ncost + self.env.heuristic(ni, nj)
                heapq.heappush(frontier, (f, (ni, nj, ncost), path + [action]))
        return None, float("inf")

if __name__ == "__main__":
    print("=== Plagiarism Detection using A* Search ===")
    doc1_path = input("Enter path to Document 1 file: ").strip()
    doc2_path = input("Enter path to Document 2 file: ").strip()
    doc1_path = clean_filepath(doc1_path)
    doc2_path = clean_filepath(doc2_path)
    doc1 = read_document_from_file(doc1_path)
    doc2 = read_document_from_file(doc2_path)
    print("\nPreprocessed Document 1:", doc1)
    print("Preprocessed Document 2:", doc2)
    env = PlagiarismEnv(doc1, doc2)
    agent = AStarAgent(env)
    path, cost = agent.a_star()
    print("\n=== Alignment Result ===")
    for step in path:
        print(step)
    print("Total alignment cost:", cost)
    print("\n=== Potential Plagiarism Sentences ===")
    threshold = 3
    plagiarized_sentences = []
    for step in path:
        if step[0] == "align":
            sent1, sent2 = step[1], step[2]
            dist = env.edit_distance(sent1, sent2)
            if dist <= threshold:
                plagiarized_sentences.append((sent1, sent2, dist))
                print(f'[Plagiarized] "{sent1}" <-> "{sent2}" (distance={dist})')
    if len(plagiarized_sentences) > 0:
        percent = (len(plagiarized_sentences) / max(len(doc1), len(doc2))) * 100
        print(f"\nFinal Result: Documents are plagiarized ({percent:.2f}% similarity)")
    else:
        print("\nFinal Result: Documents are not plagiarized")

