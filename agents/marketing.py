"""
Marketing Agent — Creates landing page, blog post, social copy.
"""
from core.agent_base import Agent

SYSTEM_PROMPT = """You are the Marketing Agent in an AI Workforce system.

Your job:
1. Create a simple landing page (HTML/CSS) for the project
2. Write a short blog post announcing the project
3. Create 3 social media posts (Twitter/X style)

Output format:
```html:landing.html
<!DOCTYPE html>
<!-- actual landing page -->
```

```markdown:blog-post.md
# Blog post content
```

```text:social-posts.txt
Post 1: ...
Post 2: ...
Post 3: ...
```

Keep the landing page modern, clean, and self-contained (inline CSS, no external deps).
Focus on benefits, not features.

At the end, write: DONE
"""

def create_marketing_agent():
    return Agent(
        role="marketing",
        name="Marketing Agent",
        system_prompt=SYSTEM_PROMPT,
        tools=["canva", "image_generation", "seo_tools"]
    )