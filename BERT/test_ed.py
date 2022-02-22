# Copyright 2018 CVTE . All Rights Reserved.
# coding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


class Solution:
    def minDistance(self, word1: str, word2: str):

        m = len(word1)
        n = len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # dp[i][j]表示word1[:i]与word2[:j]之间的编辑距离

        # 若一个字符串为空，编辑距离等于另一个字符串的长度
        for i in range(1, m + 1):
            dp[i][0] = i

        for j in range(1, n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1

        return dp

    def backpath(self, word1, word2, dp):
        i = len(dp) - 1
        j = len(dp[0]) - 1
        res = []
        while i > 0 or j > 0:
            a = dp[i - 1][j - 1] if i > 0 and j > 0 else float("inf")
            b = dp[i - 1][j] if i > 0 else float("inf")
            c = dp[i][j - 1] if j > 0 else float("inf")
            min_val = min([a, b, c])

            if dp[i][j] == a and a == min_val:
                i -= 1
                j -= 1
                # 没有操作
            elif a == min([a, b, c]):
                #  通过替换来的
                i -= 1
                j -= 1
                res.append((i, i + 1, word1[i], word2[j], "sub"))
            elif b == min([a, b, c]):
                i = i - 1
                res.append((i, i + 1, word1[i], "", "del"))
            else:
                j = j - 1
                res.append((i + 1, i + 1, "", word2[j], "ins"))
        print(res)
        return res


obj = Solution()
word1 = "horse"
word2 = "ros"
# 4


# word1 = "horse"
# word2 = "o"

dp = obj.minDistance(word1, word2)
print(dp[len(word1)][len(word2)])

res = obj.backpath(word1, word2, dp)
print(res)
