# Style-Transfer-Bot
This is my first project in programming. And now i will tell you about how it was.<br>
Check <i>requirements.txt</i> for needed packages to be installed. <br>
nst.py - NST realization <br>
async_bot.py - Bot realization
config.py - API_TOKEN


## Этап первый 
Всё началось с банального ознакомления с темой GAN-ов, Style Transfer и тем, как это вообще работает. Посмотрел лекции от DLS на эти темы.
Дальше начал читать статьи: <br>
	https://nextjournal.com/gkoehler/pytorch-neural-style-transfer <br>
	https://github.com/luanfujun/deep-photo-styletransfer <br>
  https://pytorch.org/tutorials/advanced/neural_style_tutorial.html <br>
Больше всего помогла разобраться статья с официального сайта PyTorch. Основу реализации NST я взял именно от туда.
## Этап второй
Проблемы начались сразу - были несостыковки с PyCharm и версиями библиотек, некоторые библиотеки не импортировались, и еще многое-многое.
Фиксил это долго, ошибки гуглились не всегда. Но всё получилось. <br>
Написав NST, я обернул всё в классы. GAN(дополнительное задание), к сожалению, сделать уже не успевал.
## Этап третий 
Начал делать бота, подглядывал самые разные реализации, пересмотрел много видео на ютубе и в итоге разобрался.
Начал делать тесты бота, многое менял, добавлял, убирал и чинил. <br> Решил, что фотку стиля пользователю искать неудобно - добавил свои стили.
## Этап четвертый 
Задеплоить бота пытался очень долго, сначала на Heroku был превышен лимит памяти, что я пофиксил с помощью обрезания модели VGG19. <br>
Потом были разные ошибки и по итогу задеплоить получилось, но бот почему-то не отвечал на запросы. Эта проблема осталась нерешенной до дедлайна, но я попытаюсь ее исправить после.
## Использование
Если задеплоить получится, то инструкция ниже. <br>
Теперь вы можете открыть приложение telegram, чтобы общаться в чате с ботом <a href='https://t.me/StylesTransBot'>@StylesTransBot</a>.<br>

Чтобы начать разговор, нажмите кнопку "Пуск" или введите /start. Введите /help, чтобы получить более подробное описание.<br>

Далее просто следуйте инструкциям: <br>

Вам будет предложено загрузить фотографию.<br>
Затем вам следует выбрать желаемый стиль, который будет перенесен на вашу фотографию. <br>
Наконец, вы получите результат.<br> Также, вы можете посмотреть на существующий пример данной магии, нажав кнопку "Показать пример".

<img src='https://github.com/triflt/Style-Transfer-Bot/blob/main/images/window.jpg' height='250' width='250'/> <img src='https://github.com/triflt/Style-Transfer-Bot/blob/main/images/pic.jpg' height='250' width='250'/>
<img src='https://github.com/triflt/Style-Transfer-Bot/blob/main/images/resulting.jpg' height='250' width='250'/>


