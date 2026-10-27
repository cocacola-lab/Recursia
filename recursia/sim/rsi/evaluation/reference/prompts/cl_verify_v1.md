You check one relation in a tabletop robot scene. Image 1 is the scene at the START of the task; image 2 is the scene NOW
(same fixed camera). The robot was instructed: "{{INSTRUCTION}}".

Question: in image 2, is it true that {{CLAUSE}}?

Judge only what is visible. Compare with image 1 to find the object. Containment means the object is within the container's walls
or region, not resting on its rim or next to it; "on" means supported by the named surface. If the object is hidden, ambiguous, or
you cannot tell, answer "unsure". Answer "no" only when you can see that the relation does not hold.
Return JSON: {"verdict": "yes" | "no" | "unsure", "evidence": "<one short sentence>"}.
