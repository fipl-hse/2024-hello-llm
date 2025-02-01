.. _ner-label:

NER
==============

Models
------

+--------------------------------------------------------------------------+------+-----------+
| Model                                                                    | Lang | Task      |
+==========================================================================+======+===========+
| `dslim/distilbert-NER                                                    | EN   | NER       |
| <https://huggingface.co/dslim/distilbert-NER>`__                         |      |           |
+--------------------------------------------------------------------------+------+-----------+
| `Babelscape/wikineural-multilingual-ner                                  | EN   | NER       |
| <https://huggingface.co/Babelscape/wikineural-multilingual-ner>`__       |      |           |
+--------------------------------------------------------------------------+------+-----------+

Datasets
--------

1. `Babelscape/wikineural <https://huggingface.co/Babelscape/wikineural-multilingual-ner>`__

   1. **Lang**: EN
   2. **Rows**: 11590
   3. **Preprocess**:

      1. Rename column ``ner_tags`` to ``target``.
      2. Rename column ``tokens`` to ``source``.

2. `eriktks/conll2003 <https://huggingface.co/datasets/eriktks/conll2003>`__

   1. **Lang**: EN
   2. **Rows**: 3250
   3. **Preprocess**:

      1. Rename column ``ner_tags`` to ``target``.
      2. Rename column ``tokens`` to ``source``.

Metrics
-------

-  Accuracy
