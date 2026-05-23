# Coin Partitions

Let p(n) be the number of ways to partition n into positive integer parts (order does not matter). For example, p(5) = 7:

```
5
4 + 1
3 + 2
3 + 1 + 1
2 + 2 + 1
2 + 1 + 1 + 1
1 + 1 + 1 + 1 + 1
```

**Find the least positive integer n for which p(n) is divisible by 1 000 000.**

The answer is in the tens of thousands. A naive recursive implementation will not reach it in reasonable time; an efficient recurrence is needed.
