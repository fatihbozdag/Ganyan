def update_race_results(race_date, track, results):
    """Update prediction history with actual results"""
    analyzer = RaceAnalyzer()
    
    # Find prediction for this race
    for pred in analyzer.history['predictions']:
        if pred['date'] == race_date and pred['track'] == track:
            pred['results'] = results
            pred['accuracy'] = pred['predicted_winner'] == results['winner']
            
            # Update overall accuracy
            analyzer.history['accuracy']['total'] += 1
            if pred['accuracy']:
                analyzer.history['accuracy']['correct'] += 1
            
            # Update track stats
            if track not in analyzer.history['track_stats']:
                analyzer.history['track_stats'][track] = {'predictions': 0, 'correct': 0}
            analyzer.history['track_stats'][track]['predictions'] += 1
            if pred['accuracy']:
                analyzer.history['track_stats'][track]['correct'] += 1
            
            analyzer.save_history()
            print(f"Results updated for {race_date} at {track}")
            print(f"Overall accuracy: {analyzer.history['accuracy']['correct']}/{analyzer.history['accuracy']['total']}")
            return
    
    print(f"No prediction found for {race_date} at {track}") 