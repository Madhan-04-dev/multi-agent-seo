# multi-agent-seo-blog-generator
Python-based multi-agent system for generating SEO-optimized HR blog posts

# Multi-Agent SEO Blog Generator

An intelligent system that automatically generates high-quality, SEO-optimized blog posts on trending HR topics using a multi-agent architecture.

## System Architecture

This system implements a cooperative multi-agent architecture where specialized agents work together to research, plan, generate, optimize, and review content. Each agent focuses on a specific part of the content creation process.

### Agents

1. **Research Agent**: Identifies trending HR topics and gathers comprehensive information from various sources.
2. **Content Planning Agent**: Creates a structured outline and identifies optimal SEO keywords.
3. **Content Generation Agent**: Writes the blog content based on the research and outline.
4. **SEO Optimization Agent**: Ensures the content follows SEO best practices and optimizes for target keywords.
5. **Review Agent**: Proofreads and improves the final content for readability and quality.

### Agent Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Research   │ -> │   Content   │ -> │   Content   │ -> │     SEO     │ -> │   Review    │
│    Agent    │    │  Planning   │    │ Generation  │    │ Optimization│    │    Agent    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

1. **Research Phase**: The Research Agent identifies a trending HR topic (or uses a provided topic) and gathers comprehensive information.
2. **Planning Phase**: The Content Planning Agent creates a structured outline and identifies target keywords.
3. **Generation Phase**: The Content Generation Agent writes the blog content following the outline and incorporating research findings.
4. **Optimization Phase**: The SEO Optimization Agent analyzes and enhances the content for search engines.
5. **Review Phase**: The Review Agent proofreads and finalizes the content, ensuring quality and readability.

## Technologies and Tools

- **Python 3.8+**: Primary programming language
- **OpenAI API**: Used by all agents to access advanced language models
- **Markdown**: Used for content formatting
- **JSON**: Used for structured data exchange between agents
- **dotenv**: For environment variable management
- **requests**: For HTTP requests (if needed for external data)
- **argparse**: For command-line argument parsing

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/seo-blog-generator.git
   cd seo-blog-generator
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

Generate a blog post on a trending HR topic:

```bash
python blog_generator.py
```

### Specify a Topic

Generate a blog post on a specific HR topic:

```bash
python blog_generator.py --topic "AI in HR Recruitment"
```

### Custom Output Directory

Specify a custom output directory:

```bash
python blog_generator.py --output ./my_blogs
```

## Output

The system generates three files for each blog post:

1. **Markdown file** (`.md`): The blog post in markdown format
2. **HTML file** (`.html`): The blog post in HTML format, ready for web publishing
3. **Metadata file** (`.json`): Contains metadata about the blog post (title, word count, generation timestamp, etc.)

## System Requirements

- Python 3.8 or higher
- OpenAI API key
- Internet connection (for API calls)
- Minimum 2GB RAM

## Customization

You can customize the system by:

- Modifying agent prompts in the source code
- Adjusting generation parameters (temperature, max tokens, etc.)
- Adding new agents for additional functionality
- Implementing different LLM providers

## Limitations

- Requires an OpenAI API key with access to GPT-4 or similar model
- Quality depends on the underlying LLM capabilities
- Limited fact-checking capabilities - review content before publishing
- API costs may accumulate with multiple blog generations

## Future Improvements

- Add web scraping for real-time research
- Implement image generation and suggestion
- Add automatic fact-checking against reliable sources
- Include more detailed SEO analysis and scoring
- Support for multiple content formats (articles, whitepapers, etc.)

## License

MIT

## Acknowledgments

- OpenAI for providing the API
- All contributors to the open-source libraries used in this project
