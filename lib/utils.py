def word_count(example, count, bs):
    # batch_size_fn for torchtext, when using number of words as batch size, instead
    # of number of examples as batch size, (cause example is sequential data) 
    # example: the example that's been judged whether to add to current batch
    # count: number of examples in current batch
    # bs: current effective batch size, (which is used to just wherether is isbig enough) 
    # then if example is longer than the longest example in the batch, adding it would elongate it
    # and add one example
    # then after that, if the batch size is still smaller than the prescribe batch size, add it
    max_length = bs / count if count > 0 else 0
    max_length = max(len(example.form), max_length)
    return  max_length * (count + 1)

