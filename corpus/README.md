# corpus/

Public-domain prose for `tools/measure_free_choices.py` and `tools/attack_eval.py`.
Downloaded from Project Gutenberg and stripped of its boilerplate:

| file | work |
|---|---|
| gutenberg-1342 | Austen, *Pride and Prejudice* |
| gutenberg-1976 | Doyle, *The Adventures of Sherlock Holmes* |
| gutenberg-84 | Shelley, *Frankenstein* |
| gutenberg-3464 | Wollstonecraft, *A Vindication of the Rights of Woman* |
| gutenberg-18502 | Mill, *Utilitarianism* |
| gutenberg-2701 | Melville, *Moby-Dick* |

Public domain in the United States; check your own jurisdiction.

**Bias you should know about when quoting our numbers.** This is 19th-century British and
American literary prose. It uses `towards`, `amongst`, `afterwards` and `-ise` spellings far
more than a 2026 magazine essay does, so `rho` for tier `a`/`b` is likely an *over*-estimate
for modern AmE non-fiction and an *under*-estimate for modern BrE. It also contains no
contractions at scale, so tier `c` density is understated. Run the tool on your own writing:
`filigrana table --slots-from your-essay.md`.
