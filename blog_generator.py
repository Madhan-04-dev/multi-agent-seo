import os
import json
import time
import requests
import openai
from datetime import datetime
from dotenv import load_dotenv
import argparse
import re
import random
import markdown

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class Agent:
    """Base class for all agents in the system"""
    def __init__(self, name):
        self.name = name
        
    def execute(self, *args, **kwargs):
        """Execute the agent's main functionality"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def call_llm(self, prompt, system_message=None, temperature=0.7, max_tokens=2000):
        """Call the LLM API with the given prompt"""
        if system_message is None:
            system_message = f"You are a helpful {self.name}."
    
    # Try the API call up to 3 times with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",  # You can change this to your preferred model
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print(f"Attempt {attempt+1}/{max_retries} failed. Error calling LLM API: {e}")
                print(f"Waiting {wait_time} seconds before retrying...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue
        
        # If all retries fail, return a fallback response instead of None
        print("All API call attempts failed. Using fallback response.")
        
        # Construct a basic fallback response based on the prompt
        topic_match = re.search(r"topic[:\s]+([^\.]+)", prompt, re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else "the requested topic"
        
        if "keyword" in prompt.lower():
            return f"seo, best practices, guide, tips, {topic}, how to"
        elif "outline" in prompt.lower():
            return f"# {topic.title()} Guide\n\n## Introduction\n## Key Benefits\n## Best Practices\n## Common Challenges\n## Tools and Resources\n## Conclusion"
        elif "research" in prompt.lower():
            return f"Basic information about {topic}. This is fallback content due to API issues."
        else:
            return f"Information about {topic}. This is fallback content due to API issues."

class ResearchAgent(Agent):
    """Agent responsible for researching trending HR topics and gathering information"""
    def __init__(self):
        super().__init__("Research Agent")
        
    def execute(self, topic=None):
        """Research trending HR topics and gather information"""
        
        if topic:
            print(f"Researching specific topic: {topic}")
            trending_topic = topic
        else:
            # Identify trending HR topics
            print("Identifying trending HR topics...")
            trending_topics = self._identify_trending_topics()
            trending_topic = trending_topics[0] if trending_topics else "Employee Wellness Programs"
        
        # Gather information about the selected topic
        print(f"Gathering information about: {trending_topic}")
        research_results = self._gather_topic_information(trending_topic)
        
        return {
            "topic": trending_topic,
            "research": research_results
        }
    
    def _identify_trending_topics(self):
        """Identify trending HR topics"""
        prompt = """
        Identify the top 5 trending topics in Human Resources (HR) right now.
        For each topic, provide:
        1. A clear, concise title (3-7 words)
        2. A brief explanation of why it's trending (1-2 sentences)
        
        Format your response as a JSON list of objects with 'title' and 'reason' keys.
        """
        
        response = self.call_llm(prompt, 
                               system_message="You are a research specialist in HR trends. Provide accurate, current information on trending HR topics.")
        
        try:
            # Extract JSON from response if needed
            json_pattern = r"```json(.*?)```"
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = response
                
            # Try to clean up the string for JSON parsing
            json_str = json_str.replace("'", "\"")
            topics = json.loads(json_str)
            
            return [topic["title"] for topic in topics]
        except Exception as e:
            print(f"Error parsing trending topics: {e}")
            # Return fallback topics
            return ["Remote Work Policies", "Employee Wellness Programs", "AI in HR", "DEI Initiatives", "Employee Retention Strategies"]
    
    def _gather_topic_information(self, topic):
        """Gather detailed information about a specific HR topic"""
        prompt = f"""
        I need comprehensive information about the HR topic: "{topic}".
        
        Please provide:
        1. Key statistics and data points related to this topic
        2. Current trends and developments
        3. Best practices
        4. Common challenges and solutions
        5. Future predictions
        6. Notable case studies or examples
        7. Expert opinions
        8. Relevant tools or technologies
        
        Focus on providing factual, up-to-date information that would be valuable for HR professionals.
        """
        
        response = self.call_llm(prompt, 
                               system_message="You are a research specialist in HR. Provide comprehensive, accurate information.",
                               max_tokens=3000)
        
        return response

class ContentPlanningAgent(Agent):
    """Agent responsible for creating a structured outline based on research"""
    def __init__(self):
        super().__init__("Content Planning Agent")
    
    def execute(self, research_data):
        """Create a structured outline based on research data"""
        topic = research_data["topic"]
        research = research_data["research"]
        
        print(f"Creating outline for topic: {topic}")
        
        prompt = f"""
        Based on the following research about "{topic}", create a comprehensive blog post outline:
        
        {research}
        
        Create an SEO-optimized outline that includes:
        1. A compelling title with the main keyword "{topic}"
        2. An introduction section
        3. 5-7 main sections with appropriate H2 headings
        4. 2-3 subsections (H3) under each main section where appropriate
        5. A conclusion section
        6. A call to action
        
        For each section and subsection, provide:
        - A clear, SEO-friendly heading
        - A brief 1-2 sentence description of what should be covered
        - Key points to include
        
        Format the outline as a structured markdown document.
        """
        
        outline = self.call_llm(prompt, 
                              system_message="You are a content strategist specializing in SEO blog structure. Create comprehensive, well-organized outlines.",
                              max_tokens=2500)
        
        # Generate SEO keywords for the topic
        keywords = self._generate_keywords(topic, research)
        
        return {
            "topic": topic,
            "outline": outline,
            "keywords": keywords
        }
    
    def _generate_keywords(self, topic, research):
        """Generate relevant SEO keywords based on the topic and research"""
        prompt = f"""
        Based on the HR topic "{topic}" and the following research:
        
        {research[:500]}...
        
        Generate:
        1. One primary keyword phrase (the main focus of the article)
        2. Five secondary keyword phrases (important related terms)
        3. Ten related long-tail keywords (specific phrases people might search for)
        
        Format your response as a JSON object with keys 'primary', 'secondary', and 'longtail'.
        """
        
        response = self.call_llm(prompt, 
                               system_message="You are an SEO specialist. Generate relevant, high-value keywords for HR content.")
        
        try:
            # Extract JSON from response if needed
            json_pattern = r"```json(.*?)```"
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = response
                
            # Try to clean up the string for JSON parsing
            json_str = json_str.replace("'", "\"")
            keywords = json.loads(json_str)
            
            return keywords
        except Exception as e:
            print(f"Error parsing keywords: {e}")
            # Return fallback keywords
            return {
                "primary": topic,
                "secondary": [f"{topic} best practices", f"{topic} strategies", f"{topic} trends", f"{topic} examples", f"{topic} benefits"],
                "longtail": [f"how to implement {topic}", f"benefits of {topic} for businesses", f"{topic} case studies"]
            }

class ContentGenerationAgent(Agent):
    """Agent responsible for writing the blog post based on outline and research"""
    def __init__(self):
        super().__init__("Content Generation Agent")
    
    def execute(self, content_plan, research_data):
        """Write the blog post based on outline and research"""
        topic = content_plan["topic"]
        outline = content_plan["outline"]
        keywords = content_plan["keywords"]
        research = research_data["research"]
        
        print(f"Generating blog content for: {topic}")
        
        # Extract sections from the outline
        sections = self._extract_sections(outline)
        
        # Generate content for each section
        full_content = ""
        for section in sections:
            section_content = self._generate_section_content(
                section, 
                topic, 
                keywords, 
                research
            )
            full_content += section_content + "\n\n"
        
        return {
            "topic": topic,
            "content": full_content,
            "keywords": keywords
        }
    
    def _extract_sections(self, outline):
        """Extract sections from the outline"""
        # Simple implementation - in a real system you would parse the markdown properly
        sections = []
        current_section = {"heading": "", "description": "", "subsections": []}
        
        for line in outline.split("\n"):
            line = line.strip()
            
            if line.startswith("# "):
                # Title - skip
                continue
            elif line.startswith("## "):
                # Save previous section if exists
                if current_section["heading"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "heading": line[3:],
                    "description": "",
                    "subsections": []
                }
            elif line.startswith("### "):
                # Subsection
                current_section["subsections"].append({
                    "heading": line[4:],
                    "description": ""
                })
            elif current_section["heading"] and not line.startswith("#"):
                # Add to current section description
                if current_section["subsections"]:
                    current_section["subsections"][-1]["description"] += line + " "
                else:
                    current_section["description"] += line + " "
        
        # Add the last section
        if current_section["heading"]:
            sections.append(current_section)
            
        return sections
    
    def _generate_section_content(self, section, topic, keywords, research):
        """Generate content for a specific section"""
        heading = section["heading"]
        description = section["description"]
        subsections = section["subsections"]
        
        # Construct system message with SEO guidance
        system_message = f"""
        You are an expert content writer specializing in HR topics.
        Write engaging, informative content that follows SEO best practices:
        - Naturally incorporate keywords without keyword stuffing
        - Use an engaging, professional tone appropriate for HR professionals
        - Include specific examples and actionable advice
        - Write in active voice with clear, concise language
        - Keep paragraphs short (3-5 sentences)
        - End sections with transitions to maintain flow
        """
        
        # Primary keyword to focus on for this section
        primary = keywords.get("primary", topic)
        secondary = keywords.get("secondary", [])
        longtail = keywords.get("longtail", [])
        
        # Randomly select keywords to focus on for this section
        focus_keywords = [primary]
        if secondary:
            focus_keywords.append(random.choice(secondary))
        if longtail:
            focus_keywords.append(random.choice(longtail))
        
        prompt = f"""
        Write content for the following section of a blog post about "{topic}":
        
        ## {heading}
        
        Section description: {description}
        
        Relevant research:
        {research[:800]}...
        
        Guidelines:
        1. Write approximately 250-350 words for this section
        2. Start with a strong opening paragraph that introduces the section topic
        3. Naturally incorporate these focus keywords: {', '.join(focus_keywords)}
        4. Include statistics or data points where relevant
        5. Provide practical, actionable advice
        6. End with a smooth transition to the next section
        
        Format as markdown with the section heading (H2) at the top.
        """
        
        # If there are subsections, include them in the prompt
        if subsections:
            prompt += "\n\nInclude these subsections (H3):\n"
            for subsection in subsections:
                prompt += f"### {subsection['heading']}\n{subsection['description']}\n\n"
        
        # Generate content for the section
        section_content = self.call_llm(prompt, system_message=system_message, max_tokens=1500)
        
        return section_content

class SEOOptimizationAgent(Agent):
    """Agent responsible for optimizing content for SEO"""
    def __init__(self):
        super().__init__("SEO Optimization Agent")
    
    def execute(self, generated_content):
        """Optimize content for SEO"""
        topic = generated_content["topic"]
        content = generated_content["content"]
        keywords = generated_content["keywords"]
        
        print(f"Optimizing content for SEO: {topic}")
        
        # Check SEO elements
        seo_analysis = self._analyze_seo(content, keywords)
        
        # Optimize content based on analysis
        optimized_content = self._optimize_content(content, seo_analysis, keywords)
        
        return {
            "topic": topic,
            "content": optimized_content,
            "seo_analysis": seo_analysis
        }
    
    def _analyze_seo(self, content, keywords):
        """Analyze content for SEO elements"""
        prompt = f"""
        Analyze the following blog content for SEO effectiveness:
        
        {content[:1000]}...
        
        The target keywords are:
        - Primary: {keywords.get('primary', '')}
        - Secondary: {', '.join(keywords.get('secondary', []))}
        
        Provide an SEO analysis that includes:
        1. Keyword usage assessment (frequency and placement)
        2. Heading structure evaluation
        3. Content readability assessment
        4. Internal linking opportunities
        5. Meta description recommendations
        6. Image and alt text recommendations
        
        Format your response as a JSON object with these categories as keys.
        """
        
        response = self.call_llm(prompt, 
                               system_message="You are an SEO specialist. Analyze content and provide actionable SEO recommendations.")
        
        try:
            # Extract JSON from response if needed
            json_pattern = r"```json(.*?)```"
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = response
                
            # Try to clean up the string for JSON parsing
            json_str = json_str.replace("'", "\"")
            analysis = json.loads(json_str)
            
            return analysis
        except Exception as e:
            print(f"Error parsing SEO analysis: {e}")
            # Return fallback analysis
            return {
                "keyword_usage": "Review keyword placement in headings and first/last paragraphs",
                "heading_structure": "Ensure proper H1, H2, H3 hierarchy",
                "readability": "Aim for shorter paragraphs and sentences",
                "internal_linking": "Add 3-5 internal linking opportunities",
                "meta_description": f"Include the primary keyword '{keywords.get('primary', '')}' in meta description",
                "images": "Add relevant images with keyword-rich alt text"
            }
    
    def _optimize_content(self, content, seo_analysis, keywords):
        """Optimize content based on SEO analysis"""
        prompt = f"""
        Optimize the following blog content based on this SEO analysis:
        
        SEO ANALYSIS:
        {json.dumps(seo_analysis, indent=2)}
        
        KEYWORDS:
        - Primary: {keywords.get('primary', '')}
        - Secondary: {', '.join(keywords.get('secondary', [])[:3])}
        - Long-tail examples: {', '.join(keywords.get('longtail', [])[:3])}
        
        CONTENT TO OPTIMIZE:
        {content}
        
        Make these SEO improvements:
        1. Adjust keyword placement and frequency (especially in headings, intro, and conclusion)
        2. Improve heading structure if needed
        3. Enhance readability (shorter paragraphs, bullet points where appropriate)
        4. Add internal linking opportunities (use [Link text](URL) format)
        5. Suggest a meta description
        6. Suggest image placements with alt text
        
        Return the fully optimized content in markdown format, with your meta description suggestion at the top as an HTML comment <!-- Meta: description here -->.
        """
        
        optimized_content = self.call_llm(prompt, 
                                       system_message="You are an SEO optimization specialist. Enhance content to improve search rankings while maintaining readability and value.",
                                       max_tokens=4000)
        
        return optimized_content

class ReviewAgent(Agent):
    """Agent responsible for proofreading and improving content quality"""
    def __init__(self):
        super().__init__("Review Agent")
    
    def execute(self, optimized_content):
        """Proofread and improve content quality"""
        topic = optimized_content["topic"]
        content = optimized_content["content"]
        
        print(f"Reviewing and finalizing content: {topic}")
        
        # Extract meta description from content if it exists
        meta_description = ""
        meta_match = re.search(r"<!--\s*Meta:\s*(.*?)\s*-->", content)
        if meta_match:
            meta_description = meta_match.group(1)
            # Remove meta description comment from content
            content = re.sub(r"<!--\s*Meta:\s*(.*?)\s*-->", "", content).strip()
        
        # Proofread content
        final_content = self._proofread_content(content)
        
        # Add metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        word_count = len(final_content.split())
        
        metadata = {
            "topic": topic,
            "title": self._extract_title(final_content),
            "meta_description": meta_description,
            "word_count": word_count,
            "generated_at": timestamp
        }
        
        return {
            "content": final_content,
            "metadata": metadata,
            "html": markdown.markdown(final_content)
        }
    
    def _extract_title(self, content):
        """Extract the title (H1) from the content"""
        title_match = re.search(r"^#\s+(.*?)$", content, re.MULTILINE)
        if title_match:
            return title_match.group(1)
        return "Untitled"
    
    def _proofread_content(self, content):
        """Proofread and improve content quality"""
        prompt = f"""
        Proofread and improve the following blog content:
        
        {content}
        
        Make these improvements:
        1. Fix any grammar, spelling, or punctuation errors
        2. Improve sentence structure and flow
        3. Ensure consistency in tone and style
        4. Check for and correct any factual inaccuracies
        5. Verify that all headings follow proper markdown format
        6. Ensure proper citation format if references are included
        
        Return the improved content in markdown format.
        """
        
        proofread_content = self.call_llm(prompt, 
                                       system_message="You are a professional editor and proofreader. Improve content quality while maintaining the author's voice and intent.",
                                       max_tokens=4000)
        
        return proofread_content

class BlogGeneratorSystem:
    """Main system that coordinates all agents"""
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.planning_agent = ContentPlanningAgent()
        self.generation_agent = ContentGenerationAgent()
        self.seo_agent = SEOOptimizationAgent()
        self.review_agent = ReviewAgent()
        
    def generate_blog(self, topic=None, output_dir="output"):
        """Generate a complete blog post"""
        print("Starting blog generation process...")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Step 1: Research
        research_data = self.research_agent.execute(topic)
        
        # Step 2: Content Planning
        content_plan = self.planning_agent.execute(research_data)
        
        # Step 3: Content Generation
        generated_content = self.generation_agent.execute(content_plan, research_data)
        
        # Step 4: SEO Optimization
        optimized_content = self.seo_agent.execute(generated_content)
        
        # Step 5: Review and Finalization
        final_blog = self.review_agent.execute(optimized_content)
        
        # Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = re.sub(r'[^a-zA-Z0-9]', '_', research_data["topic"].lower())
        
        # Save markdown version
        md_path = os.path.join(output_dir, f"{topic_slug}_{timestamp}.md")
        with open(md_path, "w") as f:
            f.write(final_blog["content"])
        
        # Save HTML version
        html_path = os.path.join(output_dir, f"{topic_slug}_{timestamp}.html")
        with open(html_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{final_blog["metadata"]["title"]}</title>
    <meta name="description" content="{final_blog["metadata"]["meta_description"]}">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #444; margin-top: 30px; }}
        h3 {{ color: #555; }}
        a {{ color: #0066cc; }}
        pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #ddd; padding-left: 15px; color: #666; }}
        img {{ max-width: 100%; height: auto; }}
        .metadata {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="metadata">
        <p><strong>Topic:</strong> {final_blog["metadata"]["topic"]}</p>
        <p><strong>Word Count:</strong> {final_blog["metadata"]["word_count"]}</p>
        <p><strong>Generated:</strong> {final_blog["metadata"]["generated_at"]}</p>
    </div>
    {final_blog["html"]}
</body>
</html>""")
        
        # Save metadata
        meta_path = os.path.join(output_dir, f"{topic_slug}_{timestamp}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(final_blog["metadata"], f, indent=2)
        
        print(f"Blog generation complete!")
        print(f"Markdown file saved to: {md_path}")
        print(f"HTML file saved to: {html_path}")
        print(f"Metadata saved to: {meta_path}")
        
        return {
            "markdown_path": md_path,
            "html_path": html_path,
            "metadata_path": meta_path,
            "metadata": final_blog["metadata"]
        }

def main():
    parser = argparse.ArgumentParser(description="Generate an SEO-optimized HR blog post")
    parser.add_argument("--topic", type=str, help="Specific HR topic to write about")
    parser.add_argument("--output", type=str, default="output", help="Output directory for generated blog")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        return
    
    # Create and run the blog generator system
    blog_system = BlogGeneratorSystem()
    result = blog_system.generate_blog(args.topic, args.output)
    
    print(f"\nBlog generation complete! Topic: {result['metadata']['title']}")
    print(f"Word count: {result['metadata']['word_count']}")

if __name__ == "__main__":
    main()