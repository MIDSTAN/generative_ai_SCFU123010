# Build a 2-step pipeline 
# 1. exract the cosre claim from a news article as a list.
# 2. generate a short fact card (headline, 3 bullet points, source confidence note) use only the exreacted claim from step 1.
# input article text
#  list of extracted claims from step1 and the fact card from step2

system_prompt = """
you are a factcard generator expert.

"""
