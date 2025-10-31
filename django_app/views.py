# chatbot/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .utils import BibleSearchEngine

# Initialize search engine (singleton pattern)
search_engine = BibleSearchEngine()


def index(request):
    """Render the main chatbot page"""
    return render(request, 'chatbot/index.html')


@csrf_exempt
def search_bible(request):
    """
    API endpoint for searching Bible chapters with verse-level highlighting
    POST /api/search/
    Body: {"query": "user question"}
    
    Returns:
    {
        "success": True,
        "query": "user question",
        "results": [
            {
                "book": "Genesis",
                "chapter": 1,
                "chapter_id": "Genesis 1",
                "similarity": 0.85,
                "verses": [
                    {
                        "verse": 1,
                        "text": "In the beginning...",
                        "similarity": 0.92
                    },
                    ...
                ]
            },
            ...
        ],
        "count": 5
    }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        if not query:
            return JsonResponse({'error': 'Query cannot be empty'}, status=400)
        
        # Perform multi-stage search with verse highlighting
        results = search_engine.search(query)
        
        return JsonResponse({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        import traceback
        traceback.print_exc()  # For debugging
        return JsonResponse({'error': str(e)}, status=500)