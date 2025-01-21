import tkinter as tk
from tkinter import ttk, messagebox
import json
from race_analyzer import RaceAnalyzer
import os
from datetime import datetime
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

class RaceAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Horse Race Analyzer")
        self.root.geometry("1200x800")
        
        # Initialize analyzer
        self.analyzer = RaceAnalyzer()
        
        # Create main containers
        self.create_race_info_frame()
        self.create_horses_frame()
        self.create_buttons_frame()
        
        # Load saved races if exists
        self.load_saved_races()
        
    def create_race_info_frame(self):
        """Create frame for race information"""
        frame = ttk.LabelFrame(self.root, text="Race Information", padding="10")
        frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Race info fields - add City field
        labels = ['Track:', 'City:', 'Distance (m):', 'Surface:', 'Weather:', 
                  'Temperature:', 'Class:', 'Prize:']
        self.race_info_vars = {}
        
        # Create a dictionary of common Turkish racing cities and their tracks
        self.city_tracks = {
            'İstanbul': ['Veliefendi'],
            'Ankara': ['75. Yıl'],
            'İzmir': ['Şirinyer'],
            'Bursa': ['Osmangazi'],
            'Adana': ['Yeşiloba'],
            'Elazığ': ['Elazığ'],
            'Şanlıurfa': ['GAP'],
            'Kocaeli': ['İzmit']
        }
        
        # Create combobox for city selection
        ttk.Label(frame, text='City:').grid(row=0, column=0, padx=5, pady=2)
        self.city_var = tk.StringVar()
        city_combo = ttk.Combobox(frame, textvariable=self.city_var, 
                                 values=list(self.city_tracks.keys()))
        city_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Create combobox for track selection
        ttk.Label(frame, text='Track:').grid(row=0, column=2, padx=5, pady=2)
        self.track_var = tk.StringVar()
        self.track_combo = ttk.Combobox(frame, textvariable=self.track_var)
        self.track_combo.grid(row=0, column=3, padx=5, pady=2)
        
        # Update track options when city changes
        def update_tracks(*args):
            city = self.city_var.get()
            if city in self.city_tracks:
                self.track_combo['values'] = self.city_tracks[city]
                if self.city_tracks[city]:
                    self.track_combo.set(self.city_tracks[city][0])
        
        self.city_var.trace('w', update_tracks)
        
        # Other race info fields
        row = 1
        for i, label in enumerate(labels[2:], 2):  # Skip City and Track as they're handled above
            ttk.Label(frame, text=label).grid(row=row, column=(i%2)*2, padx=5, pady=2)
            var = tk.StringVar()
            self.race_info_vars[label] = var
            ttk.Entry(frame, textvariable=var).grid(row=row, column=(i%2)*2+1, padx=5, pady=2)
            if i % 2 == 1:
                row += 1
            
    def create_horses_frame(self):
        """Create frame for horse entries"""
        frame = ttk.LabelFrame(self.root, text="Horse Entries", padding="10")
        frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # Headers
        headers = ['Name', 'Age', 'Type', 'Weight', 'Jockey', 'Trainer', 'Recent Form', 'Best Time']
        for i, header in enumerate(headers):
            ttk.Label(frame, text=header).grid(row=0, column=i, padx=5, pady=2)
        
        # Horse entry rows
        self.horse_entries = []
        for row in range(1, 13):  # Support up to 12 horses
            row_vars = {}
            for col, header in enumerate(headers):
                var = tk.StringVar()
                ttk.Entry(frame, textvariable=var).grid(row=row, column=col, padx=2, pady=1)
                row_vars[header] = var
            self.horse_entries.append(row_vars)
            
    def create_buttons_frame(self):
        """Create frame for buttons"""
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(frame, text="Save Race", command=self.save_race).grid(row=0, column=0, padx=5)
        ttk.Button(frame, text="Load Race", command=self.load_race).grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Analyze Race", command=self.analyze_race).grid(row=0, column=2, padx=5)
        ttk.Button(frame, text="Clear Form", command=self.clear_form).grid(row=0, column=3, padx=5)
        ttk.Button(frame, text="Update Results", command=self.show_results_window).grid(row=0, column=4, padx=5)
        
        # Add visualization frame
        self.viz_frame = ttk.LabelFrame(self.root, text="Analysis Results", padding="10")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
    def save_race(self):
        """Save current race details"""
        race_info = {
            'City': self.city_var.get(),
            'Track': self.track_var.get()
        }
        race_info.update({k.replace(':', ''): v.get() for k, v in self.race_info_vars.items()})
        horses = []
        
        for entry in self.horse_entries:
            name = entry['Name'].get()
            if name:  # Only save horses with names
                horse = {k: v.get() for k, v in entry.items()}
                horses.append(horse)
        
        if not horses:
            messagebox.showwarning("Warning", "No horses entered!")
            return
            
        race_data = {
            'info': race_info,
            'horses': horses,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Save to file
        os.makedirs('data/saved_races', exist_ok=True)
        filename = f"race_{race_info['Track']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f'data/saved_races/{filename}', 'w') as f:
            json.dump(race_data, f, indent=4)
            
        messagebox.showinfo("Success", "Race saved successfully!")
        
    def load_race(self):
        """Load a saved race"""
        # Show saved races in a new window
        load_window = tk.Toplevel(self.root)
        load_window.title("Load Saved Race")
        load_window.geometry("400x300")
        
        # Create listbox with saved races
        listbox = tk.Listbox(load_window, width=50)
        listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        for race in self.saved_races:
            listbox.insert(tk.END, f"{race['info']['Track']} - {race['date']}")
            
        def load_selected():
            selection = listbox.curselection()
            if selection:
                race = self.saved_races[selection[0]]
                self.load_race_data(race)
                load_window.destroy()
                
        ttk.Button(load_window, text="Load", command=load_selected).pack(pady=10)
        
    def load_race_data(self, race):
        """Load race data into form"""
        # Clear current form
        self.clear_form()
        
        # Load race info
        for k, v in race['info'].items():
            if k + ':' in self.race_info_vars:
                self.race_info_vars[k + ':'].set(v)
                
        # Load horses
        for i, horse in enumerate(race['horses']):
            if i < len(self.horse_entries):
                for k, v in horse.items():
                    self.horse_entries[i][k].set(v)
                    
    def analyze_race(self):
        """Analyze current race and show results"""
        # Gather race info
        race_info = {k.replace(':', ''): v.get() for k, v in self.race_info_vars.items()}
        
        # Gather horse entries
        race_entries = []
        for entry in self.horse_entries:
            name = entry['Name'].get()
            if name:
                horse = {
                    'name': name,
                    'age': int(entry['Age'].get() or 0),
                    'type': entry['Type'].get().lower(),
                    'weight': entry['Weight'].get(),
                    'jockey': entry['Jockey'].get(),
                    'trainer': entry['Trainer'].get(),
                    'recent_form': entry['Recent Form'].get(),
                    'best_time': entry['Best Time'].get()
                }
                race_entries.append(horse)
                
        if not race_entries:
            messagebox.showwarning("Warning", "No horses entered!")
            return
            
        # Run analysis
        predictions = self.analyzer.analyze_race(race_entries, race_info)
        
        # Show results in GUI
        self.show_analysis_results(predictions, race_info)
        
    def show_analysis_results(self, predictions, race_info):
        """Display analysis results in GUI"""
        # Clear previous visualizations
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
            
        # Create figure with subplots
        fig = plt.Figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Win Probability Chart
        ax1 = fig.add_subplot(gs[0, 0])
        horses = [p['horse_name'] for p in predictions]
        probabilities = [p['win_chance'] for p in predictions]
        ax1.barh(horses, probabilities)
        ax1.set_title('Win Probabilities')
        ax1.set_xlabel('Probability (%)')
        
        # 2. Recent Form Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        form_data = []
        for p in predictions:
            positions = [int(pos) for pos in p['recent_form'].split() if pos.isdigit()]
            form_data.append(positions[-6:] if len(positions) > 6 else positions)
        ax2.imshow(form_data, cmap='RdYlGn_r')
        ax2.set_title('Recent Form')
        ax2.set_yticks(range(len(horses)))
        ax2.set_yticklabels(horses)
        
        # 3. Race Details
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        details = [
            f"City: {race_info['City']}",
            f"Track: {race_info['Track']}",
            f"Distance: {race_info['Distance']}m",
            f"Surface: {race_info['Surface']}",
            f"Weather: {race_info['Weather']}",
            f"Temperature: {race_info['Temperature']}",
            f"Class: {race_info['Class']}",
            f"Prize: {race_info['Prize']}"
        ]
        ax3.text(0.5, 0.5, '\n'.join(details), 
                ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Embed in GUI
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    def clear_form(self):
        """Clear all form fields"""
        for var in self.race_info_vars.values():
            var.set('')
        for entry in self.horse_entries:
            for var in entry.values():
                var.set('')
                
    def load_saved_races(self):
        """Load previously saved races"""
        self.saved_races = []
        try:
            for filename in os.listdir('data/saved_races'):
                if filename.endswith('.json'):
                    with open(f'data/saved_races/{filename}', 'r') as f:
                        self.saved_races.append(json.load(f))
        except FileNotFoundError:
            pass

    def show_results_window(self):
        """Show window to update race results"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Update Race Results")
        results_window.geometry("500x600")
        
        # Create frame for results
        frame = ttk.Frame(results_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Headers
        ttk.Label(frame, text="Horse").grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(frame, text="Position").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(frame, text="Time").grid(row=0, column=2, padx=5, pady=2)
        
        # Entry fields for each horse
        result_entries = []
        row = 1
        for entry in self.horse_entries:
            name = entry['Name'].get()
            if name:
                ttk.Label(frame, text=name).grid(row=row, column=0, padx=5, pady=2)
                
                pos_var = tk.StringVar()
                ttk.Entry(frame, textvariable=pos_var, width=10).grid(row=row, column=1, padx=5, pady=2)
                
                time_var = tk.StringVar()
                ttk.Entry(frame, textvariable=time_var, width=10).grid(row=row, column=2, padx=5, pady=2)
                
                result_entries.append({
                    'name': name,
                    'position': pos_var,
                    'time': time_var
                })
                row += 1
        
        def save_results():
            results = {
                'winner': None,
                'positions': {},
                'times': {}
            }
            
            for entry in result_entries:
                pos = entry['position'].get()
                if pos:
                    pos = int(pos)
                    results['positions'][entry['name']] = pos
                    if pos == 1:
                        results['winner'] = entry['name']
                    
                time = entry['time'].get()
                if time:
                    results['times'][entry['name']] = time
            
            if results['winner']:
                race_info = {k.replace(':', ''): v.get() for k, v in self.race_info_vars.items()}
                from update_results import update_race_results
                update_race_results(
                    datetime.now().strftime('%Y-%m-%d'),
                    race_info['Track'],
                    results
                )
                messagebox.showinfo("Success", "Results updated successfully!")
                results_window.destroy()
            else:
                messagebox.showwarning("Warning", "Please specify the winner (position 1)")
        
        ttk.Button(frame, text="Save Results", command=save_results).grid(
            row=row, column=0, columnspan=3, pady=20)

def main():
    root = tk.Tk()
    app = RaceAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 