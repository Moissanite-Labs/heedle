# Reviewing changes to Heedle

A change is ready to merge when:

1. **It does what it says.** The description matches the diff.
2. **A test fails without it,** where the change is to behaviour. For a change to the mathematics, both languages change and the vectors are rebuilt.
3. **CI is green on the exact head.**
4. **Claims match the code.** A document changed by the pull request says nothing the tests do not check, or it marks the statement as unchecked. Statements about other repositories are checked like any factual claim.
5. **The marks in the theory are honest.** [P] for what follows from the definitions, [T] for what a named test checks on a toy host, [K] for a known result, [I] for a reading.
6. **Every commit is signed off.**
7. **It holds one change.**

Not reasons to block: style, a disagreement about an idea in the roadmap, or an open question left open. Raise these as comments.

The author does not merge. Use a merge commit when the order of the commits is part of the evidence.

## What does not belong here

Experiment records, audits and design notes go to [weft-loom](https://github.com/Moissanite-Labs/weft-loom) and are reviewed by its lanes. A pull request here that reports a measurement should link the record instead of carrying it.
