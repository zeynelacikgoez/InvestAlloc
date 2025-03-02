import os

def schreibe_datei_inhalte(start_verzeichnis, output_datei):
    output_datei_abs = os.path.abspath(output_datei)
    script_abs = os.path.abspath(__file__)
    
    with open(output_datei, 'w', encoding='utf-8') as out_file:
        # Durchlaufe rekursiv alle Unterordner und Dateien
        for root, dirs, files in os.walk(start_verzeichnis):
            for datei in files:
                # Überspringe .DS_Store-Dateien
                if datei == '.DS_Store':
                    continue
                
                dateipfad = os.path.join(root, datei)
                dateipfad_abs = os.path.abspath(dateipfad)
                
                # Überspringe das Output-File und das Skript selbst
                if dateipfad_abs in {output_datei_abs, script_abs}:
                    continue
                
                # Schreibe den Header mit Pfad und Dateiname
                out_file.write(f"#### {dateipfad} ####\n\n")
                out_file.write("inhalt: ")
                try:
                    # Versuche den Inhalt der Datei zu lesen
                    with open(dateipfad, 'r', encoding='utf-8') as in_file:
                        inhalt = in_file.read()
                    # Schreibe den Inhalt in Anführungszeichen
                    out_file.write(f'"{inhalt}"\n\n')
                except Exception as e:
                    # Falls ein Fehler beim Lesen auftritt, schreibe eine Fehlermeldung
                    out_file.write(f"Fehler beim Lesen der Datei: {e}\n\n")

if __name__ == "__main__":
    start_verzeichnis = os.getcwd()
    output_datei = "alle_datei_inhalte.txt"
    schreibe_datei_inhalte(start_verzeichnis, output_datei)
