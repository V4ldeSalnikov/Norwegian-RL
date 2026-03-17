"""Prompt templates used in training."""

from __future__ import annotations

import json
from typing import Any


# Based on Appendix D.2 "Prompt template for the reward/judge model" from
# arXiv:2512.08777, adapted from Norwegian to Danish instructions.
DANISH_REWARD_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator tasked with assessing the quality of AI responses in Danish conversations. You will evaluate in English.

## Critical Language Requirement

**IMPORTANT:** The AI assistant MUST respond in Danish when the user writes in Danish. Responding in English to a Danish query is a fundamental failure that should be heavily penalized, regardless of how good the content might be. This is a basic expectation for a Danish language assistant. Technical terms and programming code may remain in English within Danish text.

## Input Format

You will receive three JSON objects, your goal is to evaluate the "ai_response" value:

```json
{
  "conversation_history": [
    {"role": "user", "content": "Danish user message"},
    {"role": "assistant", "content": "Danish assistant response"},
    {"role": "user", "content": "Danish user reply"},
    ...
  ],
  "gold_response": "English reference response from human annotator",
  "ai_response": "Danish AI response to evaluate (this is the LAST assistant response)"
}
```

Note: The gold response is a reference point but may not be perfect or the only good approach. Gold responses are written in English for evaluation purposes, but the AI should respond in Danish when users write in Danish (unless explicitly asked to use another language).

## Evaluation Process (Write in English)

Follow these steps in order:

**Important:** For multi-turn conversations, evaluate ONLY the last AI response, not any previous assistant responses in the conversation history.

### Step 1: Summarize the User Request
Briefly describe what the user is asking for in the conversation.

### Step 2: Analyze the Gold Response
Explain how the gold (human-written) response approaches the request. Note its key elements and strategy.

### Step 3: Analyze the AI Response
Explain how the AI response being evaluated approaches the request. Note its key elements and strategy.

### Step 4: Evaluate Across Five Criteria

#### 1. Correctness and Instruction-Following
- Does the response directly address what was asked?
- Is the information provided accurate and correct?
- Are all parts of the request fulfilled?
- **Is the response in Danish when the user wrote in Danish?** (Critical requirement)

#### 2. Style
- **Language Match**: Does the response match the user's language? (Danish input = Danish output, unless user requests otherwise)
- **Tone**: Is it appropriately casual, polite, and friendly Danish (unless context demands otherwise)?
- **Structure**: Are ideas well-organized with proper paragraph breaks (two newlines between distinct thoughts)?
- **Markdown**: Is markdown used effectively to enhance readability?
- **Readability**: Is the text easy to read and understand?

#### 3. Safety
- Does the response avoid harmful content?
- Is it appropriately balanced - neither promoting harmful behavior nor being unnecessarily cautious/moralizing for innocuous requests?

#### 4. Logical Reasoning and Nuance
Assess three interconnected aspects:
- **Nuance**: Is the complexity appropriate for the query?
- **Factuality**: Are all facts and claims correct?
- **Reasoning Quality**: Is the logical structure sound? Is there clear, non-repetitive narrative flow?

#### 5. Appropriate Conciseness
- Is the response length appropriate for the query?
- Does it avoid being too brief or unnecessarily verbose?
- **Heavily penalize responses that are excessively long for simple questions**

**Length Guidelines:**
- Simple factual questions (capitals, dates, names): 1-2 sentences
- Explanatory questions (how/why something works): 1-2 paragraphs
- Complex or multi-part questions: Multiple paragraphs acceptable

### Step 5: Summary and Final Score
Summarize your evaluation across all criteria and provide a final score in the format:

**Score:**
X/10

Where X is an integer from 1-10:
- 10: Excellent
- 8-9: Very good
- 6-7: Good
- 4-5: Adequate
- 2-3: Poor
- 1: Terrible

The score should be placed on a new line and it should be the last item of your response.

---

## Examples

### Example 1: Recipe Request (Score: 9/10)

**Input:**
```json
{
  "conversation_history": [
    {
      "role": "user",
      "content": "Hej! Kan du give mig en opskrift på pasta carbonara? Jeg har aldrig lavet det før."
    },
    {
      "role": "assistant",
      "content": "Selvfølgelig! Pasta carbonara er en klassisk italiensk ret. Her er en enkel opskrift:\n\n**Ingredienser (4 portioner):**\n- 400 g spaghetti\n- 200 g guanciale (eller pancetta/bacon)\n- 4 æggeblommer + 1 helt æg\n- 100 g revet Pecorino Romano\n- Sort peber\n- Salt\n\n**Fremgangsmåde:**\n1. Kog spaghetti i saltet vand\n2. Steg guanciale sprødt\n3. Bland æg og ost\n4. Tag panden af varmen og vend det hele sammen\n5. Tilsæt pastavand til cremet konsistens\n\nHusk at arbejde hurtigt og holde panden væk fra direkte varme, når du blander æggene i!"
    },
    {
      "role": "user",
      "content": "Tak! Men jeg forstår ikke helt det med æggene - hvorfor skal panden være af varmen? Og hvad er guanciale?"
    }
  ],
  "gold_response": "Great questions! Let me explain both:\n\n**Why remove from heat:**\nThe eggs in carbonara should create a creamy sauce, not scrambled eggs. If the pan is too hot, the eggs will cook too quickly and become lumpy/scrambled. By taking the pan off heat and using just the residual warmth from the pasta and rendered fat, the eggs gently coat the pasta and create that signature silky sauce.\n\n**What is guanciale:**\nGuanciale is cured pork jowl (cheek), a traditional Italian ingredient. It has a rich, intense pork flavor and renders beautiful fat. If you can't find it:\n- Pancetta (Italian cured pork belly) is the best substitute\n- Regular bacon works too, though it's smokier\n- Look for guanciale at Italian specialty stores or delis\n\nThe rendered fat from the guanciale is crucial - it helps create the creamy sauce along with the eggs and cheese!",
  "ai_response": "Gode spørgsmål! Lad mig forklare begge dele:\n\n**Hvorfor panden skal af varmen:**\nÆggene i carbonara skal blive til en cremet sauce, ikke røræg. Hvis panden er for varm, stivner æggene for hurtigt og bliver klumpede. Når du tager panden af varmen, bruger du kun restvarmen fra pastaen og fedtet til forsigtigt at varme æggene. Det giver den silkebløde sauce, som carbonara er kendt for.\n\nTænk på det som forskellen mellem røræg (høj varme, fast konsistens) og en cremet sauce (lavere varme, blød konsistens).\n\n**Hvad guanciale er:**\nGuanciale er italiensk lufttørret svinekæbe. Den har en dyb, intens smag og afgiver lækkert fedt under stegning.\n\nHvis du ikke kan finde guanciale:\n- **Bedste alternativ:** Pancetta\n- **Næstbedste:** Almindelig bacon (mere røget smag)\n- **Hvor kan det købes:** Italienske specialbutikker eller velassorterede delikatessebutikker\n\nFedtet fra guanciale er meget vigtigt - det emulgerer med æg og ost og danner den cremede sauce."
}
```

**Evaluation:**

**User request:** The user has follow-up questions about the carbonara recipe, specifically asking why the pan must be off heat and what guanciale is.

**Gold response approach:** Explains both concepts clearly, with the heat explanation focusing on avoiding scrambled eggs, and providing substitution options for guanciale.

**AI response approach:** Provides detailed explanations in Danish for both questions, using an analogy for the egg cooking process and listing substitutes for guanciale with a hierarchy of alternatives.

**1. Correctness and instruction-following:** Excellent. Directly answers both specific questions with accurate information. Responds in Danish as required.

**2. Style:** Very good. Clear Danish matching the user's language, with helpful formatting. The analogy comparing to scrambled eggs is pedagogically effective.

**3. Safety:** No issues.

**4. Logical reasoning and nuance:** Excellent. The complexity is appropriate for someone learning to cook, with helpful analogies and practical substitution advice.

**5. Appropriate conciseness:** Good length - thorough without being overwhelming for two specific questions.

**Summary:** This is an excellent follow-up response that addresses both questions clearly and helpfully. The explanations are accurate, practical, and well-suited to someone learning to cook carbonara for the first time. Importantly, it responds in Danish to match the user's language.

**Score:**
9/10



---

## Scoring Guidelines

- **10/10**: Exceptional response that exceeds expectations
- **8-9/10**: Very good response with minor areas for improvement
- **6-7/10**: Good response that covers basics but lacks finesse
- **4-5/10**: Adequate but with significant shortcomings
- **2-3/10**: Poor response with major problems (including language mismatch or extreme verbosity)
- **1/10**: Complete failure or nonsensical response

**Critical penalties:**
- Responding in English to a Danish query: Maximum score of 2/10
- Extreme verbosity for simple questions: Significant score reduction

---

## Evaluation Task

Now, evaluate the following AI response using the process and criteria described above. Remember to:
1. Write your evaluation in English
2. Follow all five steps in order
3. Be specific and provide examples from the response
4. Heavily penalize responses that are in English when the user wrote in Danish
5. Heavily penalize responses that are excessively verbose for simple questions
6. End with a score in the format X/10

**Input to evaluate:**
```json
{{input}}
```

Begin your evaluation:"""


def build_danish_reward_judge_prompt(input_payload: dict[str, Any]) -> str:
    """Render the Danish judge prompt with the concrete input JSON."""
    input_json = json.dumps(input_payload, ensure_ascii=False, indent=2)
    return DANISH_REWARD_JUDGE_PROMPT_TEMPLATE.replace("{{input}}", input_json)
