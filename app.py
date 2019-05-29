from flask import Flask, render_template, request
import csv
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import treetaggerwrapper as ttpw
tagger = ttpw.TreeTagger(TAGLANG='de', TAGDIR="/Users/Asti/Documents/Projects/bon-prix/tag-cloud/TreeTagger")

# Configure application
app = Flask(__name__)

# Routing
@app.route('/')
def index():
    # # register a new dialect to define custom delimiter that is not ',' (so ';' is considered as column separator)
    csv.register_dialect('myDialect', delimiter = ';')

    # # read csv data into dictionary
    # with open('./Datensatz_Coding_Challenge.csv', 'r', encoding='latin', errors='replace') as csvFile:
    #     reader = csv.DictReader(csvFile, dialect='myDialect')
    #     dataDict = []
    #     for row in reader:
    #         dataDict.append(dict(row))
    # csvFile.close()

    dataDict = [
         {'StyleID': '515928', 'text': 'Super Passform ', 'rating': '5'}, 
         {'StyleID': '654563', 'text': 'sehr angenehmer Stoff fällt aber zu Klein aus deshalb musste ich die BH zurück schicken', 'rating': '4'}, 
         {'StyleID': '655046', 'text': 'Passform, angenehm zu tragen, Preis alles bestens. ', 'rating': '5'}, 
         {'StyleID': '8623725', 'text': 'Wirklich top  Ich hätte nicht gedacht, dass ich zu dem Preis einen solchen Badeanzug bekommen kann. Er sitzt perfekt, macht schlank und kaschiert kleine Pölsterchen. Auch das Material ist o.k. Jetzt kann die Schwimmsaison starten.', 'rating': '5'}, 
         {'StyleID': '553018', 'text': 'Diese ballerinas sind einfach nur toll .habe sie schn in mehreren Farben bestellt weil sie einfach nur bequem sind .', 'rating': '4'}, {'StyleID': '709229', 'text': 'Schönes Kleid für den Urlaub. Bikini an, Kleid drüber und ab zum Strand.', 'rating': '4'}, 
         {'StyleID': '9743730', 'text': 'Alles zufriedenstellend.', 'rating': '4'}, 
         {'StyleID': '515928', 'text': 'Einwandfrei, angenehm und Preis top! ', 'rating': '5'}, 
         {'StyleID': '515928', 'text': 'Hübscher Slip, Farbe leider noch eine Nuance dunkler als auf dem Foto. Verglichen mit Premium-Marken prima Qualität zum günstigen Preis.', 'rating': '5'}, 
         {'StyleID': '553018', 'text': 'schöne Ballerinas zu einem tollen Preis , mir gefallen sie sehr gut und passen perfekt , wenn ich noch eine Frischesohle einlege. Habe nämlich normalerweise Schuhgröße 38,5 .', 'rating': '5'}, 
         {'StyleID': '434886', 'text': 'Leider gefällt mir die Passform nicht. Es ist als Mann nicht sehr angenehm wenn bei der kleinsten Bewegung die Hose nur noch zur hälfte das Gesäß bedeckt. Das bei verschiedenen Größen die ich ausprobiert habe.', 'rating': '1'}, 
         {'StyleID': '709229', 'text': 'Der Schnitt wäre sehr schön, aber die Farbe war mir zu blass.', 'rating': '3'}, 
         {'StyleID': '515928', 'text': 'Sitz gut ohne einzuschneiden, schummelt ein bisschen die pölsterchen', 'rating': '4'}, 
         {'StyleID': '655046', 'text': 'Sport- BH sehr gut, meine Frau träge nur noch diese BHs, sehr leich, ohne drückende Bügel und denoch robust und sehr Preiswert. Die Bestellung kam sehr schnell, nach 2 Tagen, super, bin sehr zufrieden!', 'rating': '5'}, 
         {'StyleID': '8623725', 'text': 'Material Schnitt und Farbe sind ok, was für mich garnicht geht sind die eingenähten Softcups laut beschreibung sollen sie herausnehmbar sein, was leider aber fest vernäht waren, selbst wenn man sie herausnehmen könnte ist das oderteil  für kleine Brüste viel zu groß, daher zurück.', 'rating': '3'}, 
         {'StyleID': '8623725', 'text': 'Gut sitzender Badeanzug. Die Raffung am Bauch ist wirklich sehr vorteilhaft! Verstellbare Träger würden das Gesamtbild perfekt machen. ', 'rating': '4'}, 
         {'StyleID': '709229', 'text': 'Wunderschönes Kleid.Passt sehr gut.Leider trotz eingehaltener Waschanleitung total verfärbt.Reklamation per E-Mail bei bonprix blieb bisher unbeantwortet.', 'rating': '3'}, 
         {'StyleID': '9743730', 'text': 'Gut', 'rating': '4'}, {'StyleID': '1709054', 'text': 'Alles perfekt, sehr angenehmes Material trägt sich sehr gut. ', 'rating': '5'}, 
         {'StyleID': '9743730', 'text': 'Sehr gute Qualität, dunkelt 100% ab, sieht gut aus.', 'rating': '5'}, 
         {'StyleID': '8623725', 'text': '50 zu groß/weit und 48 zu kurz. Beide zurück. ', 'rating': '1'}, 
         {'StyleID': '515928', 'text': 'angenehm zu tragen super Qualität passt perfekt', 'rating': '5'}, 
         {'StyleID': '434886', 'text': 'Qualität und Farbe ist gut. Hosenbund hinten ist zu niedrig im Vergleich zu vorne.', 'rating': '2'}, 
         {'StyleID': '1709054', 'text': 'Schlimme Fetzen! Hatten sie vor der Vorab-Wäsche noch Passform, so ist diese danach völlig abhanden gekommen. Lapprig, total verzogen, der Stoff klebt aneinander und die Farben sind nicht so kräftig, wie abgebildet. Geblieben sind unförmige Hängerchen. Normalerweise müsste man so was zurückschicken. Auch wenn man den sehr günstigen Preis betrachtet, so sind sie diesen nicht wert. Bin seit den Anfängen Kundin und hatte die Shirts schon damals gekauft .... kein Vergleich und eins lebt sogar noch.', 'rating': '1'}, 
         {'StyleID': '8623725', 'text': 'der Ausschnitt an den Beinen ist viel zu Groß ', 'rating': '1'}, 
         {'StyleID': '515928', 'text': 'sehr guter sitz, angenehm zu tragen..... Ich kauf jetzt alle anderen farben....', 'rating': '5'}, 
         {'StyleID': '9743730', 'text': 'Leider habe ich die falsche Größe bestellt. Nun bestelle ich die richtige Größe, da die Vorhänge sehr gut zu meiner Einrichtung passen und das Material sich gut anfühlt. Ob es die Eigenschaften erfüllt, die beschrieben sind, werde ich erst später erfahren.', 'rating': '4'}, 
         {'StyleID': '8623725', 'text': 'Der badeanzug passte in gr.50 überhaupt nicht.der beinausschnitt ist vieeeel zu gross...und oben im brustbereich sehr nach unten ziehend.', 'rating': '1'}
    ]
    
    # sentences and words
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    for row in dataDict:
        row['sentences'] = sent_tokenize(row['text'], language='german')
        row['words'] = []
        for sentence in row['sentences']:
            row['words'].extend(tokenizer.tokenize(sentence))
        # remove stopwords
        stopWords = set(stopwords.words('german'))
        row['filteredSentence'] = [w for w in row['words'] if not w in stopWords]
        
        # part of speech tagging (POS) and lemmatization
        # lemmatization was used instead of stemming, as stemming does not necessarily result in an actual word
        # whereas lemma is an actual language word
        # https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
        row['taggedSentence'] = tagger.tag_text(row['filteredSentence'])
        row['taggedSentence2'] = ttpw.make_tags(row['taggedSentence'], allow_extra=True)
        
        # Count lemmas of each product
        # 
        for tag in row['taggedSentence2']:
            print(tag)

        # print(row['taggedSentence2'])
    

        
    
    #print(dataDict)
    # print(dataDict[-2])

    dictionary = {
        'StyleID': '9743730', 
        'text': 'Leider habe ich die falsche Größe bestellt. Nun bestelle ich die richtige Größe, da die Vorhänge sehr gut zu meiner Einrichtung passen und das Material sich gut anfühlt. Ob es die Eigenschaften erfüllt, die beschrieben sind, werde ich erst später erfahren.', 
        'rating': '4', 
        'sentences': ['Leider habe ich die falsche Größe bestellt.', 'Nun bestelle ich die richtige Größe, da die Vorhänge sehr gut zu meiner Einrichtung passen und das Material sich gut anfühlt.', 'Ob es die Eigenschaften erfüllt, die beschrieben sind, werde ich erst später erfahren.'], 
        'words': ['Leider', 'habe', 'ich', 'die', 'falsche', 'Größe', 'bestellt', '.', 'Nun', 'bestelle', 'ich', 'die', 'richtige', 'Größe', ',', 'da', 'die', 'Vorhänge', 'sehr', 'gut', 'zu', 'meiner', 'Einrichtung', 'passen', 'und', 'das', 'Material', 'sich', 'gut', 'anfühlt', '.', 'Ob', 'es', 'die', 'Eigenschaften', 'erfüllt', ',', 'die', 'beschrieben', 'sind', ',', 'werde', 'ich', 'erst', 'später', 'erfahren', '.'], 
        'filteredSentence': ['Leider', 'falsche', 'Größe', 'bestellt', '.', 'Nun', 'bestelle', 'richtige', 'Größe', ',', 'Vorhänge', 'gut', 'Einrichtung', 'passen', 'Material', 'gut', 'anfühlt', '.', 'Ob', 'Eigenschaften', 'erfüllt', ',', 'beschrieben', ',', 'erst', 'später', 'erfahren', '.']
        # 'taggedSentence': ['Leider\tADV\tleider', 'falsche\tADJA\tfalsch', 'Größe\tNN\tGröße', 'bestellt\tVVPP\tbestellen', 
        #     '.\t$.\t.', 'Nun\tADV\tnun', 'bestelle\tVVFIN\tbestellen', 'richtige\tADJA\trichtig', 'Größe\tNN\tGröße', 
        #     ',\t$,\t,', 'Vorhänge\tNN\tVorhang', 'gut\tADJD\tgut', 'Einrichtung\tNN\tEinrichtung', 'passen\tVVFIN\tpassen',
        #     'Material\tNN\tMaterial', 'gut\tADJD\tgut', 'anfühlt\tVVFIN\tanfühlen', '.\t$.\t.', 'Ob\tKOUS\tob', 'Eigenschaften\tNN\tEigenschaft',
        #     'erfüllt\tVVPP\terfüllen', ',\t$,\t,', 'beschrieben\tVVPP\tbeschreiben', ',\t$,\t,', 'erst\tADV\terst', 'später\tADJD\tspät', 
        #     'erfahren\tVVPP\terfahren', '.\t$.\t.']
        # 'taggedSentence2': [Tag(word='Leider', pos='ADV', lemma='leider'), Tag(word='falsche', pos='ADJA', lemma='falsch'), 
        # Tag(word='Größe', pos='NN', lemma='Größe'), Tag(word='bestellt', pos='VVPP', lemma='bestellen'), 
        # Tag(word='.', pos='$.', lemma='.'), Tag(word='Nun', pos='ADV', lemma='nun'), Tag(word='bestelle', pos='VVFIN', lemma='bestellen'), 
        # Tag(word='richtige', pos='ADJA', lemma='richtig'), Tag(word='Größe', pos='NN', lemma='Größe'), Tag(word=',', pos='$,', lemma=','), 
        # Tag(word='Vorhänge', pos='NN', lemma='Vorhang'), Tag(word='gut', pos='ADJD', lemma='gut'), Tag(word='Einrichtung', pos='NN', lemma='Einrichtung'), 
        # Tag(word='passen', pos='VVFIN', lemma='passen'), Tag(word='Material', pos='NN', lemma='Material'), 
        # Tag(word='gut', pos='ADJD', lemma='gut'), Tag(word='anfühlt', pos='VVFIN', lemma='anfühlen'), 
        # Tag(word='.', pos='$.', lemma='.'), Tag(word='Ob', pos='KOUS', lemma='ob'), Tag(word='Eigenschaften', pos='NN', lemma='Eigenschaft'), 
        # Tag(word='erfüllt', pos='VVPP', lemma='erfüllen'), Tag(word=',', pos='$,', lemma=','), 
        # Tag(word='beschrieben', pos='VVPP', lemma='beschreiben'), Tag(word=',', pos='$,', lemma=','), 
        # Tag(word='erst', pos='ADV', lemma='erst'), Tag(word='später', pos='ADJD', lemma='spät'), 
        # Tag(word='erfahren', pos='VVPP', lemma='erfahren'), Tag(word='.', pos='$.', lemma='.')]
        }

    return render_template('index.html')

#TODO change this for production! (debug=True only for dev mode)
if __name__ == '__main__':
    app.run(debug=True)

