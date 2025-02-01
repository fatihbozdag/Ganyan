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
import seaborn as sns

class EnhancedRaceAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Horse Race Analyzer Pro")
        self.root.geometry("1400x900")
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TLabel', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10))
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        
        # Initialize analyzer
        self.analyzer = RaceAnalyzer()
        
        # Create main containers with grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.create_race_info_frame()
        self.create_horses_frame()
        self.create_analysis_frame()
        self.create_buttons_frame()
        
        # Load saved races if exists
        self.load_saved_races()
        
    def create_race_info_frame(self):
        """Create enhanced frame for race information"""
        frame = ttk.LabelFrame(self.root, text="Race Information", padding="10")
        frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Race info fields with improved layout
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
        
        # Create info grid
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X, expand=True)
        
        # City and Track selection (row 0)
        ttk.Label(info_frame, text='City:', style='Header.TLabel').grid(row=0, column=0, padx=5, pady=2)
        self.city_var = tk.StringVar()
        city_combo = ttk.Combobox(info_frame, textvariable=self.city_var, values=list(self.city_tracks.keys()))
        city_combo.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(info_frame, text='Track:', style='Header.TLabel').grid(row=0, column=2, padx=5, pady=2)
        self.track_var = tk.StringVar()
        self.track_combo = ttk.Combobox(info_frame, textvariable=self.track_var)
        self.track_combo.grid(row=0, column=3, padx=5, pady=2)
        
        # Race details (row 1-2)
        self.race_info_vars = {}
        labels = ['Distance (m):', 'Surface:', 'Weather:', 'Temperature:', 'Class:', 'Prize:']
        
        for i, label in enumerate(labels):
            row = (i // 3) + 1
            col = (i % 3) * 2
            
            ttk.Label(info_frame, text=label, style='Header.TLabel').grid(row=row, column=col, padx=5, pady=2)
            var = tk.StringVar()
            self.race_info_vars[label] = var
            ttk.Entry(info_frame, textvariable=var).grid(row=row, column=col+1, padx=5, pady=2)
        
        # Update track options when city changes
        def update_tracks(*args):
            city = self.city_var.get()
            if city in self.city_tracks:
                self.track_combo['values'] = self.city_tracks[city]
                if self.city_tracks[city]:
                    self.track_combo.set(self.city_tracks[city][0])
        
        self.city_var.trace('w', update_tracks)
        
    def create_horses_frame(self):
        """Create enhanced frame for horse entries"""
        frame = ttk.LabelFrame(self.root, text="Horse Entries", padding="10")
        frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Headers with improved styling
        headers = ['Name', 'Age', 'Type', 'Weight', 'Jockey', 'Trainer', 'Recent Form', 'Best Time']
        for i, header in enumerate(headers):
            ttk.Label(scrollable_frame, text=header, style='Header.TLabel').grid(row=0, column=i, padx=5, pady=2)
        
        # Horse entry rows with alternating colors
        self.horse_entries = []
        for row in range(1, 15):  # Support up to 14 horses
            row_vars = {}
            for col, header in enumerate(headers):
                var = tk.StringVar()
                entry = ttk.Entry(scrollable_frame, textvariable=var)
                entry.grid(row=row, column=col, padx=2, pady=1, sticky="ew")
                row_vars[header] = var
            self.horse_entries.append(row_vars)
        
        # Pack the canvas and scrollbar
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
    def create_analysis_frame(self):
        """Create enhanced frame for analysis results"""
        self.viz_frame = ttk.LabelFrame(self.root, text="Analysis Results", padding="10")
        self.viz_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
    def create_buttons_frame(self):
        """Create enhanced frame for buttons"""
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Create buttons with improved styling
        buttons = [
            ("Save Race", self.save_race),
            ("Load Race", self.load_race),
            ("Analyze Race", self.analyze_race),
            ("Clear Form", self.clear_form),
            ("Update Results", self.show_results_window)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(frame, text=text, command=command, style='TButton')
            btn.grid(row=0, column=i, padx=10)
            
    def show_analysis_results(self, predictions, race_info):
        """Display enhanced analysis results in GUI"""
        # Clear previous visualizations
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
            
        # Create figure with improved layout
        fig = plt.Figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Set style
        plt.style.use('seaborn')
        
        # 1. Win Probability Chart (Left)
        ax1 = fig.add_subplot(gs[0, 0])
        horses = [p['horse_name'] for p in predictions]
        probabilities = [p['win_chance'] for p in predictions]
        bars = ax1.barh(horses, probabilities)
        ax1.set_title('Win Probabilities', fontsize=10, pad=15)
        ax1.set_xlabel('Probability (%)')
        
        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center')
        
        # 2. Recent Form Heatmap (Middle)
        ax2 = fig.add_subplot(gs[0, 1:3])
        form_data = []
        for p in predictions:
            positions = [int(pos) for pos in p['recent_form'].split() if pos.isdigit()]
            form_data.append(positions[-6:] if len(positions) > 6 else positions)
        
        im = ax2.imshow(form_data, cmap='RdYlGn_r')
        ax2.set_title('Recent Form (Last 6 Races)', fontsize=10, pad=15)
        ax2.set_yticks(range(len(horses)))
        ax2.set_yticklabels(horses)
        
        # Add position labels
        for i in range(len(form_data)):
            for j in range(len(form_data[i])):
                ax2.text(j, i, str(form_data[i][j]), ha='center', va='center')
        
        # 3. Race Details (Bottom Left)
        ax3 = fig.add_subplot(gs[1, 0])
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
        
        # 4. Top Contenders Table (Bottom Middle-Right)
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax4.axis('off')
        
        # Sort horses by win probability
        sorted_predictions = sorted(predictions, key=lambda x: x['win_chance'], reverse=True)[:5]
        
        # Create table data
        table_data = []
        for i, p in enumerate(sorted_predictions, 1):
            table_data.append([
                f"{i}.",
                p['horse_name'],
                f"{p['win_chance']:.1f}%",
                p['recent_form'],
                p.get('best_time', 'N/A')
            ])
        
        table = ax4.table(
            cellText=table_data,
            colLabels=['Rank', 'Horse', 'Win %', 'Form', 'Best Time'],
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Adjust layout and embed in GUI
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def save_race(self):
        """Save current race information and horse entries to a file"""
        race_data = {
            'race_info': {
                'city': self.city_var.get(),
                'track': self.track_var.get(),
                **{k.replace(':', ''): v.get() for k, v in self.race_info_vars.items()}
            },
            'horses': []
        }
        
        # Collect horse entries
        for entry in self.horse_entries:
            horse_data = {k: v.get() for k, v in entry.items() if v.get().strip()}
            if horse_data:  # Only add if there's data
                race_data['horses'].append(horse_data)
        
        if not race_data['horses']:
            messagebox.showwarning('Warning', 'No horse data to save!')
            return
        
        # Create save directory if it doesn't exist
        os.makedirs('data/saved_races', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/saved_races/race_{timestamp}.json'
        
        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(race_data, f, indent=4)
            messagebox.showinfo('Success', f'Race saved successfully to {filename}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save race: {str(e)}')

    def load_saved_races(self):
        """Load previously saved races"""
        self.saved_races = []
        if os.path.exists('data/saved_races'):
            for file in os.listdir('data/saved_races'):
                if file.endswith('.json'):
                    self.saved_races.append(os.path.join('data/saved_races', file))

    def load_race(self):
        """Load a previously saved race"""
        if not self.saved_races:
            messagebox.showinfo('Info', 'No saved races found')
            return

        # Create a simple dialog to select a race file
        dialog = tk.Toplevel(self.root)
        dialog.title('Select Race')
        dialog.geometry('300x400')

        listbox = tk.Listbox(dialog)
        listbox.pack(fill=tk.BOTH, expand=True)

        for race_file in self.saved_races:
            listbox.insert(tk.END, os.path.basename(race_file))

        def load_selected():
            selection = listbox.curselection()
            if not selection:
                return
            
            file_path = self.saved_races[selection[0]]
            try:
                with open(file_path, 'r') as f:
                    race_data = json.load(f)

                # Load race info
                self.city_var.set(race_data['race_info']['city'])
                self.track_var.set(race_data['race_info']['track'])
                for key, var in self.race_info_vars.items():
                    clean_key = key.replace(':', '')
                    if clean_key in race_data['race_info']:
                        var.set(race_data['race_info'][clean_key])

                # Load horse entries
                self.clear_form()  # Clear existing entries
                for i, horse in enumerate(race_data['horses']):
                    if i < len(self.horse_entries):
                        for key, value in horse.items():
                            if key in self.horse_entries[i]:
                                self.horse_entries[i][key].set(value)

                dialog.destroy()
                messagebox.showinfo('Success', 'Race loaded successfully')
            except Exception as e:
                messagebox.showerror('Error', f'Failed to load race: {str(e)}')

        load_btn = ttk.Button(dialog, text='Load', command=load_selected)
        load_btn.pack(pady=10)

    def analyze_race(self):
        """Analyze the current race data"""
        # Collect race info
        race_info = {
            'City': self.city_var.get(),
            'Track': self.track_var.get(),
            'Distance': self.race_info_vars['Distance (m):'].get(),
            'Surface': self.race_info_vars['Surface:'].get(),
            'Weather': self.race_info_vars['Weather:'].get(),
            'Temperature': self.race_info_vars['Temperature:'].get(),
            'Class': self.race_info_vars['Class:'].get(),
            'Prize': self.race_info_vars['Prize:'].get()
        }

        # Collect horse entries
        race_entries = []
        for entry in self.horse_entries:
            horse_data = {k.lower(): v.get().strip() for k, v in entry.items() if v.get().strip()}
            if horse_data:
                race_entries.append(horse_data)

        if not race_entries:
            messagebox.showwarning('Warning', 'No horse data to analyze!')
            return

        try:
            # Analyze race using the analyzer
            predictions = self.analyzer.analyze_race(race_entries, race_info)
            # Display results
            self.show_analysis_results(predictions, race_info)
        except Exception as e:
            messagebox.showerror('Error', f'Analysis failed: {str(e)}')

    def clear_form(self):
        """Clear all form fields"""
        # Clear race info
        self.city_var.set('')
        self.track_var.set('')
        for var in self.race_info_vars.values():
            var.set('')

        # Clear horse entries
        for entry in self.horse_entries:
            for var in entry.values():
                var.set('')

    def show_results_window(self):
        """Show a window with detailed race results"""
        # Create a new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title('Race Results')
        results_window.geometry('800x600')

        # Add a text widget to display results
        text_widget = tk.Text(results_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(results_window, orient='vertical', command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)

        # Insert race information
        text_widget.insert(tk.END, 'Race Information:\n', 'heading')
        text_widget.insert(tk.END, f"City: {self.city_var.get()}\n")
        text_widget.insert(tk.END, f"Track: {self.track_var.get()}\n")
        for key, var in self.race_info_vars.items():
            text_widget.insert(tk.END, f"{key} {var.get()}\n")

        text_widget.insert(tk.END, '\nHorse Entries:\n', 'heading')
        for entry in self.horse_entries:
            horse_data = {k: v.get().strip() for k, v in entry.items() if v.get().strip()}
            if horse_data:
                text_widget.insert(tk.END, '\n')
                for key, value in horse_data.items():
                    text_widget.insert(tk.END, f"{key}: {value}\n")

        # Make the text widget read-only
        text_widget.configure(state='disabled')

def main():
    root = tk.Tk()
    app = EnhancedRaceAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()