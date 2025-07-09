import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.writer_agent import WriterAgent

def main():
    api_key = os.getenv("API_KEY")  # Reads API key from environment variable

    agent = WriterAgent(api_key=api_key)

    game_info = {
        "date": "2025-07-08",
        "venue": "Wembley Stadium",
        "score": {"Team A": 2, "Team B": 1}
    }

    team_info = {
        "home": {"name": "Team A"},
        "away": {"name": "Team B"}
    }

    player_info = {
        "key_player": "Player 2",
        "performance": "Scored the winning goal and assisted the equalizer"
    }

    research = {
        "storylines": [
            "A dramatic comeback in the second half.",
            "Player 2 was instrumental in the win.",
            "Team A now sits at the top of the league table."
        ],
        "quotes": [
            "Coach John: 'This team never gives up. They showed their spirit today.'",
            "Player 2: 'I just gave my all for the badge.'"
        ]
    }

    try:
        article = agent.generate_article(game_info, team_info, player_info, research)
        print("\n✅ Generated Article:\n")
        print(article)

        # Save as plain text
        with open("generated_article.txt", "w", encoding="utf-8") as f:
            f.write(article)
        print("\n📄 Article saved to 'generated_article.txt'.")

        # Convert to HTML and save
        html_article = f"""
        <html>
        <head>
            <title>Football Recap Article</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #1a1a1a; }}
                p {{ line-height: 1.6; }}
            </style>
        </head>
        <body>
            <h1>{article.splitlines()[0]}</h1>
            {"".join([f"<p>{line}</p>" for line in article.splitlines()[1:] if line.strip()])}
        </body>
        </html>
        """

        with open("generated_article.html", "w", encoding="utf-8") as f:
            f.write(html_article)
        print("\n🌐 HTML version saved to 'generated_article.html'.")

        # Export to PDF using pdfkit
        try:
            import pdfkit
            pdfkit.from_file("generated_article.html", "generated_article.pdf")
            print("\n📄 PDF version saved to 'generated_article.pdf'.")
        except ImportError:
            print("\n⚠️ pdfkit not installed. Skipping PDF export.")
        except Exception as e:
            print(f"\n❌ Error exporting PDF: {e}")

    except Exception as e:
        print(f"\n❌ Error generating article: {e}")

if __name__ == "__main__":
    main()
